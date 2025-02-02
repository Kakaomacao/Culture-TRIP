import re
import json
from time import sleep
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser

from utils.custom_wiki import CustomWikipediaAPI
from iterative_refinement.prompt_templates import prompt1, prompt2, prompt3, prompt4, prompt5

# Load environment variables (API keys)
from dotenv import load_dotenv
load_dotenv()

class GraphState(TypedDict):
    culture_noun: str
    prompts: list[str]
    wikisearch_context: str
    googlesearch_context: str
    score_history: dict
    score: dict
    feedback: str
    augmented_prompt: list[str]
    refine_recur_counter: int

# Initialize the LLM
llm = ChatOllama(model="llama3:70b")

# Initialize search tools for Wikipedia and Google
wiki_search = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())
google_search = GoogleSearchAPIWrapper()

llm_chain1 = prompt1 | llm | StrOutputParser()
llm_chain2 = prompt2 | llm | StrOutputParser()
llm_chain3 = prompt3 | llm | StrOutputParser()
llm_chain4 = prompt4 | llm | StrOutputParser()
llm_chain5 = prompt5 | llm | StrOutputParser()

GRAPH_SLEEP_TIME = 0
GRPAH_RESULT_SHOW = True

# ---------------------------------------------------------
# Load culture nouns
# ---------------------------------------------------------

def culture_noun_load(state: GraphState) -> GraphState:
    state['augmented_prompt'] = ''
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### GRAPH START ###")
        print("culture_noun:", state['culture_noun'])
    
    score_history = {
        'Clarity': [],
        'Visual_detail': [],
        'Background': [],
        'Purpose': [],
        'Comparable_object': [],
        'Total_score': []
    }
    
    sleep(GRAPH_SLEEP_TIME)    
    return GraphState(culture_noun = state['culture_noun'], score_history=score_history, refine_recur_counter = 0, aug_recur_counter = 0)

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

def refiner(state: GraphState) -> GraphState:
    information = state["wikisearch_context"] + '\n' + state["googlesearch_context"]
    
    # refine 모델
    response = llm_chain1.invoke({
        "culture_noun":state["culture_noun"],
        "information":information,
        "context":state["context"],
        "feedback": state["feedback"], 
    })
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### REFINER ###\n" + response)
    
    sleep(GRAPH_SLEEP_TIME)   
    
    return GraphState(context=response)

def evaluator(state: GraphState) -> GraphState:
    # 점수 채점 모델
    response = llm_chain2.invoke({
        "culture_noun":state["culture_noun"],
        "context":state["context"],
    })

    print("\n\n### EVALUATOR ###") if GRPAH_RESULT_SHOW else None

    # 정규 표현식을 사용하여 JSON 부분 추출
    json_match = re.search(r'\{.*?\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        score = json.loads(json_str)
        
        # 히스토리 업데이트
        score_history = state['score_history']        
        for key in score_history.keys():
            score_history[key].append(score[key])
            
        print(f'score : {score}') if GRPAH_RESULT_SHOW else None
    else:
        print("JSON data not found") 

    sleep(GRAPH_SLEEP_TIME)   
    
    return GraphState(score=score, score_history=score_history)

def feedbacker(state: GraphState) -> GraphState:
    counter = state["refine_recur_counter"] + 1
    
    # 피드백 생성 모델
    response = llm_chain3.invoke({
        "culture_noun":state["culture_noun"],
        "context":state["context"],
        "score":state["score"]
    })
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### FEEDBACKER ###")
        print(response)
    
    sleep(GRAPH_SLEEP_TIME)   
    
    return GraphState(feedback=response, refine_recur_counter=counter)

def check_feedback(state: GraphState) -> GraphState:
    feedback_score = state['score'] # 고민 필요
    total = feedback_score["Total_score"]
    counter = state["refine_recur_counter"]
    THRESH = 40
    # THRESH = 24
    
    if total >= THRESH or counter >= 5:
        return "sufficient"
    else:
        return "insufficient"
