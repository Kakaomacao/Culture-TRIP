import re
import json
from time import sleep
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser

from utils.graph_workflow import GraphState
from iterative_refinement.customWiki import CustomWikipediaAPI
from iterative_refinement.prompt_templates import prompt1, prompt2, prompt3, prompt4, prompt5
from main import batch_size

# Initialize the LLM
llm = ChatOllama(model="llama3:70b")

# Initialize search tools for Wikipedia and Google
wsearch = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())
gsearch = GoogleSearchAPIWrapper()


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

def cultural_noun_load(state: GraphState) -> GraphState:
    state['augmented_prompt'] = ''
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### GRAPH START ###")
        print("keyword:", state['cultural_noun'])
    
    score_history = {
        'Clarity': [],
        'Visual_detail': [],
        'Background': [],
        'Purpose': [],
        'Comparable_object': [],
        'Total_score': []
    }
    
    sleep(GRAPH_SLEEP_TIME)    
    return GraphState(cultural_noun = state['cultural_noun'], score_history=score_history, refine_recur_counter = 0, aug_recur_counter = 0)

# ---------------------------------------------------------
# Retrieve information
# ---------------------------------------------------------

def retrieve_info(cultural_noun):
    """
    Perform searches on Wikipedia and Google for the given cultural noun.

    Args:
        cultural_noun (str): The cultural noun to search for.

    Returns:
        tuple: A tuple containing the Wikipedia search context and Google search context.
    """
    wikisearch_context = wsearch.run(cultural_noun)
    googlesearch_context = gsearch.run("What is " + cultural_noun)
    return wikisearch_context, googlesearch_context

# ---------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------

def refiner(state: GraphState) -> GraphState:
    information = state["wikisearch_context"] + '\n' + state["googlesearch_context"]
    feedback = state["feedback"]
    
    # refine 모델
        
    response = llm_chain1.invoke({
        "cultural_noun":state["cultural_noun"],
        "category":state["category"],
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
        "cultural_noun":state["cultural_noun"],
        "category":state["category"],
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
    context = state["context"]
    keyword = state["cultural_noun"]
    score = state["score"]
    counter = state["refine_recur_counter"] + 1
    
    # 피드백 생성 모델
    
    response = llm_chain3.invoke({
        "cultural_noun":state["cultural_noun"],
        "category":state["category"],
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

# ---------------------------------------------------------
# Augmentation
# ---------------------------------------------------------

def keyword_prompt(state: GraphState) -> GraphState:
    prompts = state["prompts"]
    cultural_noun = state["cultural_noun"]
    formatted_prompts = []
    
    for prompt in prompts:
        # 중괄호 안의 문자열을 추출하기 위한 정규 표현식
        matches = re.findall(r'\{(.*?)\}', prompt)
        
        # 각 문자열을 해당하는 포맷 데이터로 대체
        for match in matches:
            prompt = prompt.replace(f'{{{match}}}', cultural_noun)

        formatted_prompts.append(prompt)
    
        
    if GRPAH_RESULT_SHOW:
        print("\n\n### ADD KEYWORD ###")
        print(formatted_prompts)    
    
    return GraphState(prompts=formatted_prompts)

def llm_augment_prompt(state: GraphState) -> GraphState:    
    pass
    print("\n\n### AUGMENTATION ###") if GRPAH_RESULT_SHOW else None

    prompts = state["prompts"]
    total_aug = []
    
    for i in range(0, len(prompts), batch_size):
        aug_recur_counter = 0
        aug_fail = True
        
        if (len(prompts) - i) <= batch_size:
            # 마지막 루프 처리
            batch_prompts = prompts[i:]
            current_batch = len(prompts) - i
        else:
            batch_prompts = prompts[i:i + batch_size]
            current_batch = batch_size
        
        while aug_recur_counter < 5 and aug_fail:
            aug_recur_counter += 1
            
            response = llm_chain5.invoke({
                "cultural_noun":state["cultural_noun"], 
                "context":state["context"], # wikisearch_context로 바꿔서 결과 받고 이미지 생성
                "prompts":batch_prompts
            })
            
            # 정규 표현식을 사용하여 리스트 부분 추출
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            
            if list_match:
                try:
                    list_str = list_match.group(0)
                    augmented_prompt = json.loads(list_str)
                    
                    aug_fail = False if len(augmented_prompt) == current_batch else True
                    
                    if GRPAH_RESULT_SHOW:
                        print(f'augmented_prompt : {augmented_prompt}')
                        print('증강 개수:', len(augmented_prompt), ', 현재 배치:', current_batch)
                except:
                    augmented_prompt = []
                    print('Augmented prompt not found')
            else:
                print(f'response ==> {response}')
                print(f'list_match ==> {list_match}')
                augmented_prompt = []
                print("List data not found")
                
                with open('error.txt', 'a') as error_file:
                    error_file.write('------------------------\n')
                    error_file.write(f'response: {response}\n')
                    error_file.write(f'list_match: {list_match}\n')
                    error_file.write('------------------------\n')
        # while 종료
        
        total_aug.extend(augmented_prompt)
        
    state["wikisearch_context"] = ''
    state["googlesearch_context"] = ''
    state["context"] = ''

    return GraphState(augmented_prompt=total_aug)