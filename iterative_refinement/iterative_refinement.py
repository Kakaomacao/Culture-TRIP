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

class GraphState(TypedDict):
    culture_noun: str
    prompt: list[str]
    refined_prompt: list[str]
    wikisearch_context: str
    googlesearch_context: str
    score_history: dict
    score: dict
    feedback: str
    refine_recur_counter: int
    is_intermediate_result_show: bool

# Initialize the LLM
llm = ChatOllama(model="llama3:70b")

# Initialize search tools for Wikipedia and Google
wiki_search = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())
google_search = GoogleSearchAPIWrapper()

refine_llm = refine_prompt | llm | StrOutputParser()
scoring_llm = scoring_prompt | llm | StrOutputParser()
feedback_llm = feedback_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# Load culture nouns
# ---------------------------------------------------------

def culture_noun_load(state: GraphState) -> GraphState:
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
    
    return GraphState(culture_noun = state['culture_noun'], score_history=score_history, feedback='', refine_recur_counter = 0, aug_recur_counter = 0)

# ---------------------------------------------------------
# Retrieve information
# ---------------------------------------------------------

def retrieve_info(state: GraphState) -> GraphState:
    culture_noun = state['culture_noun']
    wikisearch_context = wiki_search.run(culture_noun)
    googlesearch_context = google_search.run("What is " + culture_noun)

    return GraphState(wikisearch_context = wikisearch_context, googlesearch_context=googlesearch_context)

# ---------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------

def refine(state: GraphState) -> GraphState:
    information = state["wikisearch_context"] + '\n' + state["googlesearch_context"]
    
    # refine 모델
    response = refine_llm.invoke({
        "culture_noun":state["culture_noun"],
        "information":information,
        "prompt":state["prompt"],
        "feedback": state["feedback"], 
    })
    
    if state['is_intermediate_result_show']:
        print("\n\n### REFINER ###\n" + response)
    
    return GraphState(refined_prompt=response)

def scoring(state: GraphState) -> GraphState:
    # 점수 채점 모델
    response = scoring_llm.invoke({
        "culture_noun":state["culture_noun"],
        "prompt":state["prompt"],
    })

    print("\n\n### SCORING ###") if state['is_intermediate_result_show'] else None

    # 정규 표현식을 사용하여 JSON 부분 추출
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        score = json.loads(json_str)
        
        # 히스토리 업데이트
        score_history = state['score_history']        
        for key in score_history.keys():
            score_history[key].append(score[key])
            
        print(f'score : {score}') if state['is_intermediate_result_show'] else None
    else:
        print("JSON data not found") 

    return GraphState(score=score, score_history=score_history)

def feedback(state: GraphState) -> GraphState:
    counter = state["refine_recur_counter"] + 1
    
    # 피드백 생성 모델
    response = feedback_llm.invoke({
        "culture_noun":state["culture_noun"],
        "refined_prompt":state["refined_prompt"],
        "score":state["score"]
    })
    
    if state['is_intermediate_result_show']:
        print("\n\n### FEEDBACKER ###")
        print(response)
        
    return GraphState(feedback=response, refine_recur_counter=counter)

def check_score(state: GraphState) -> GraphState:
    THRESH = 40
    
    score = state['score'] # 고민 필요
    total = score["Total_score"]
    counter = state["refine_recur_counter"]
    
    if total >= THRESH or counter >= 5:
        return "sufficient"
    else:
        return "insufficient"
