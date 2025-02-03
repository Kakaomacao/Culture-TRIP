import re
import json
from time import sleep
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser

from utils.custom_wiki import CustomWikipediaAPI
from iterative_refinement.prompt_templates import refine_prompt, scoring_prompt, feedback_prompt

# Load environment variables (API keys)
from dotenv import load_dotenv
load_dotenv()

# Define the GraphState data structure to store processing state
class GraphState(TypedDict):
    culture_noun: str
    prompt: list[str]
    refined_prompt: list[str]
    wikisearch_context: str
    googlesearch_context: str
    score_history: dict
    score: dict
    feedback: str
    score_threshold: int
    refine_recur_counter: int
    is_intermediate_result_show: bool

# Initialize the language model (LLM)
llm = ChatOllama(model="llama3:70b")

# Initialize search tools for Wikipedia and Google
wiki_search = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())
google_search = GoogleSearchAPIWrapper()

# Define refinement, scoring, and feedback pipelines using LLM
refine_llm = refine_prompt | llm | StrOutputParser()
scoring_llm = scoring_prompt | llm | StrOutputParser()
feedback_llm = feedback_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# Load culture noun and initialize state
# ---------------------------------------------------------
def culture_noun_load(state: GraphState) -> GraphState:
    """
    Initializes the state with a culture noun and resets refinement tracking.
    """
    if state['is_intermediate_result_show']:
        print("\n\n### GRAPH START ###")
        print("prompt:", state['prompt'])
        print("culture_noun:", state['culture_noun'])

    score_history = {
        'Clarity': [],
        'Visual_detail': [],
        'Background': [],
        'Purpose': [],
        'Comparable_object': [],
        'Total_score': []
    }
    
    return GraphState(culture_noun=state['culture_noun'], refined_prompt='', feedback='', score_history=score_history, refine_recur_counter=0)

# ---------------------------------------------------------
# Retrieve contextual information from Wikipedia and Google
# ---------------------------------------------------------
def retrieve_info(state: GraphState) -> GraphState:
    """
    Retrieves contextual information about the given culture noun from Wikipedia and Google.
    """
    culture_noun = state['culture_noun']
    wikisearch_context = wiki_search.run(culture_noun)
    googlesearch_context = google_search.run("What is " + culture_noun)

    return GraphState(wikisearch_context=wikisearch_context, googlesearch_context=googlesearch_context)

# ---------------------------------------------------------
# Iterative refinement process
# ---------------------------------------------------------
def refine(state: GraphState) -> GraphState:
    """
    Uses an iterative refinement model to improve the prompt based on contextual information.
    """
    information = state["wikisearch_context"] + '\n' + state["googlesearch_context"]
    
    response = refine_llm.invoke({
        "culture_noun": state["culture_noun"],
        "information": information,
        "prompt": state["prompt"],
        "refined_prompt": state["refined_prompt"],
        "feedback": state["feedback"], 
    })
    
    if state['is_intermediate_result_show']:
        print("\n\n### REFINER ###\n" + response)
    
    return GraphState(refined_prompt=response)

# ---------------------------------------------------------
# Score the refined prompt
# ---------------------------------------------------------
def scoring(state: GraphState) -> GraphState:
    """
    Evaluates the refined prompt and assigns a score based on predefined criteria.
    """
    response = scoring_llm.invoke({
        "culture_noun": state["culture_noun"],
        "refined_prompt": state["refined_prompt"],
    })

    print("\n\n### SCORING ###") if state['is_intermediate_result_show'] else None

    json_match = re.search(r'\{.*?\}', response, re.DOTALL)  # Extract JSON content

    if json_match:
        json_str = json_match.group(0)
        score = json.loads(json_str)
        
        # Update score history
        score_history = state['score_history']        
        for key in score_history.keys():
            score_history[key].append(score[key])
            
        print(f'score : {score}') if state['is_intermediate_result_show'] else None
    else:
        print("JSON data not found") 

    return GraphState(score=score, score_history=score_history)

# ---------------------------------------------------------
# Generate feedback for further refinement
# ---------------------------------------------------------
def feedback(state: GraphState) -> GraphState:
    """
    Generates feedback based on the refined prompt and current scores to improve the prompt further.
    """
    counter = state["refine_recur_counter"] + 1
    
    response = feedback_llm.invoke({
        "culture_noun": state["culture_noun"],
        "refined_prompt": state["refined_prompt"],
        "score": state["score"]
    })
    
    if state['is_intermediate_result_show']:
        print("\n\n### FEEDBACKER ###")
        print(response)
        
    return GraphState(feedback=response, refine_recur_counter=counter)

# ---------------------------------------------------------
# Check whether the prompt meets the threshold score
# ---------------------------------------------------------
def check_score(state: GraphState) -> GraphState:
    """
    Checks if the refined prompt meets the score threshold. If not, further refinement is needed.
    """
    score = state['score'] 
    total = score["Total_score"]
    counter = state["refine_recur_counter"]
    
    if total >= state['score_threshold'] or counter >= 5:
        return "sufficient"
    else:
        return "insufficient"
