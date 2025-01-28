#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated
import operator

# GraphState 상태를 저장하는 용도로 사용합니다.
class GraphState(TypedDict):
    cultural_noun: str  # 문화적 고유 명사 
    category: str # 고유 명사 카테고리
    prompts: list[str] # 증강 타겟
    wikisearch_context: str # 위키피디아 검색 결과
    googlesearch_context: str # 구글 검색 결과
    context: str # 종합된 답변 결과
    score_history: dict # 피드백 점수 히스토리
    score: dict # 피드백 점수
    feedback: str # 피드백
    augmented_prompt: list[str] # 증강된 프롬프트
    refine_recur_counter: int # cycle 돈 횟수

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from tqdm import tqdm

llm = ChatOllama(model="llama3:70b")

# --------------------------------------------------------------------------------------------------------------------------------

# Template1 : Writing a prompt about the culture noun with information
template1 = """### Instruction:
I want you to write 2 ~ 3 detailed sentences with INFORMATION.
Please revise the CONTEXT content by referring to the INFORMATION as noted in the FEEDBACK.
There may be incorrect information in the INFORMATION, so be cautious and ensure it pertains to the KEYWORD before using it.

### TODO:
KEYWORD:
{cultural_noun}

INFORMATION:
{information}

CONTEXT:
{context}

FEEDBACK:
{feedback}

Answer:
"""
prompt1 = PromptTemplate.from_template(template1)

llm_chain1 = prompt1 | llm | StrOutputParser()


# Template2 : Evaluating the prompt with 5 criteria
template2 = """### Instruction:
Please evalutate CONTEXT with 5 criteria (Clarity, Detail, Context, Purpose, Comparable object).
Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the keywords.
Visual detail: Whether the prompt provides a sufficient amount of visual information, such as colors, shapes, etc.
Background: Whether the historical or temporal background information provided in the prompt is appropriate.
Purpose: Whether the description of the intended use or the users of the subject in the prompt is appropriate.
Comparable object: How well the prompt compares to existing well-known or popular examples.
Each criteria can not exceed score 10.
I want each score of criteria and total score.
Please answer with the format below.

ANSWER FORMAT:
{{   
    "Clarity" : 5,
    "Visual_detail" : 5,
    "Background" : 5,
    "Purpose" : 5,
    "Comparable_object" : 5,
    "Total_score" : 25
}}

### TODO:
KEYWORD:
{cultural_noun}

CONTEXT:
{context}

Answer:
"""
prompt2 = PromptTemplate.from_template(template2)

llm_chain2 = prompt2 | llm | StrOutputParser()


# Template3 : Feedback on the prompt
template3 = """### Instruction:
Review the items of SCORE and provide feedback on how to improve each item's score, specifically focusing on the modification of CONTEXT about KEYWORD.

### TODO:
KEYWORD:
{cultural_noun}

CONTEXT:
{context}

SCORE:
{score}

Answer:
"""

prompt3 = PromptTemplate.from_template(template3)

llm_chain3 = prompt3 | llm | StrOutputParser()


# Template4 : Adding keyword to the prompt
template4 = """### Instruction:
The KEYWORD and BASE PROMPTS are related to the CATEGORY. 
In each of the Base Prompts, there is a mask coverd with {{ }} and replace them with KEYWORD including {{}}.

Please answer in the FORMAT below. I want to have an ANSWER in format of list.
ANSWER FORMAT:
["sentence", "sentence", ...]

### TODO:
KEYWORD:
{cultural_noun}

Base Prompts:
{prompts}

Answer:
"""

prompt4 = PromptTemplate.from_template(template4)

llm_chain4 = prompt4 | llm | StrOutputParser()


# Template5 : Augmenting the prompt
template5 = """### Instruction:
Use all the information from the CONTEXT to add adtional information to each sentence of the Base prompts. Explain it so well that someone who doesn't know the KEYWORD can draw a picture of the KEYWORD just by reading the Base prompts. 

Refer to the following criteria:
Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the keywords.
Visual detail: Whether the prompt provides a sufficient amount of visual information, such as colors, shapes, etc.
Background: Whether the historical or temporal background information provided in the prompt is appropriate.
Purpose: Whether the description of the intended use or the users of the subject in the prompt is appropriate.
Comparable object: How well the prompt compares to existing well-known or popular examples.

If a Base prompt cannot accommodate the KEYWORD, allow for slight modifications to ensure all sentences are covered. 
When adding additional information to a single sentence to provide sufficient detail, expand the original 1 sentence into 3 sentences so that the final result is approximately 300 characters long.

Please answer in the FORMAT below. I want to have an ANSWER in format of list.
ANSWER FORMAT:
["3 sentences", "3 sentences", ...]


### TODO:
KEYWORD:
{cultural_noun}

CONTEXT:
{context}

Base Prompts:
{prompts}

Answer:
"""

prompt5 = PromptTemplate.from_template(template5)

llm_chain5 = prompt5 | llm | StrOutputParser()

# --------------------------------------------------------------------------------------------------------------------------------

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.utils import format_searched_web, format_searched_docs
from rag.customWiki import CustomWikipediaAPI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
import ast
import json
from time import sleep
import re

GRAPH_SLEEP_TIME = 0
GRPAH_RESULT_SHOW = True

# 위키피디아 서치 설정
wsearch = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())

# 구글 서치 설정
gsearch = GoogleSearchAPIWrapper()

# 문화적 고유명사 불러오기
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

# 웹에서 문서를 검색하여 관련성 있는 정보를 찾습니다.
def retrieve_info(state: GraphState) -> GraphState:
    # 위키피디아에서 검색하여 관련성 있는 데이터를 찾습니다.
    wikisearch_context = wsearch.run(state["cultural_noun"])
    googlesearch_context = gsearch.run("What is " + state["cultural_noun"])
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### RETRIEVE INFO ###\n<wiki>"+wikisearch_context+'\n<google>'+googlesearch_context)

    sleep(GRAPH_SLEEP_TIME)    
    # 얻어낸 정보를 wikisearch_context 키에 저장합니다.
    return GraphState(wikisearch_context=wikisearch_context, googlesearch_context=googlesearch_context)

# 정보들을 종합하고 정제합니다. Feedback도 반영합니다.
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
    

# CoT를 사용해 얻은 정보를 확인하고 피드백합니다.
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

# 종합된 정보 피드백
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

# 프롬프트에 키워드를 추가합니다.
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

# 피드백 확인
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

# LLM을 사용하여 프롬프트를 증강합니다.
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

# In[11]:


from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# langgraph.graph에서 StateGraph와 END를 가져옵니다.
workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("문화적 고유 명사", cultural_noun_load) 
workflow.add_node("정보 retrieve", retrieve_info) 
workflow.add_node("정보 refine", refiner)
workflow.add_node("정보 평가", evaluator)
workflow.add_node("피드백", feedbacker)
workflow.add_node("프롬프트 키워드 추가", keyword_prompt)  
workflow.add_node("프롬프트 강화", llm_augment_prompt)  


# 각 노드들을 연결합니다.
workflow.add_edge("문화적 고유 명사", "정보 retrieve")
workflow.add_edge("정보 retrieve", "정보 refine")
workflow.add_edge("정보 refine", "정보 평가")
workflow.add_conditional_edges(
    "정보 평가",
    check_feedback,
    {
        "sufficient": "프롬프트 키워드 추가",
        "insufficient": "피드백"
    }
) 
workflow.add_edge("피드백", "정보 refine")
workflow.add_edge("프롬프트 키워드 추가", "프롬프트 강화") 
workflow.add_edge("프롬프트 강화", END) 


# 시작 노드
workflow.set_entry_point("문화적 고유 명사")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


# In[12]: => 그림 그리는 부분은 생략


# from IPython.display import Image, display

# try:
#     display(
#         Image(app.get_graph(xray=True).draw_mermaid_png())
#     )  # 실행 가능한 객체의 그래프를 mermaid 형식의 PNG로 그려서 표시합니다. xray=True는 추가적인 세부 정보를 포함합니다.
# except:
#     print(1)
#     # 이 부분은 추가적인 의존성이 필요하며 선택적으로 실행됩니다.
#     pass


# In[13]:


import os
import json



# JSON 파일 읽기 함수
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 모든 JSON 파일 읽기
def read_all_json_files(folder_path):
    all_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            country_name = file_name.split('.')[0]
            all_data[country_name] = read_json(file_path)
    return all_data

# prompts 디렉토리에서 txt 파일 읽기 - 수정된 부분
def get_prompts_from_directory(directory, categories, file_names):
    category_prompts = {category: [] for category in categories}
    for category, file_name in zip(categories, file_names):
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                prompts = [line.strip() for line in lines]
                if category in category_prompts:
                    category_prompts[category].extend(prompts)
    return category_prompts

# 각 나라의 데이터에서 문장 생성
def generate_sentences_from_all_data(all_data, category_prompts):
    sentences = []
    for country, data in all_data.items():
        for category, items in data.items():
            if category in category_prompts:
                for item in items:
                    sentence = {
                        "cultural_noun": item,
                        "category": category,
                        "prompts": category_prompts[category],
                        "country": country
                    }
                    sentences.append(sentence)
    return sentences

# app.invoke 함수 호출
def app_invoke(cultural_noun, category, prompts, country):
    global output_path
    
    inputs = {
        "cultural_noun": cultural_noun,
        "category": category,
        "prompts": prompts
    }
    print(f'{category} : {cultural_noun} of {country}')
    output = app.invoke(inputs, config={"configurable": {"thread_id": 1111}, "recursion_limit": 100})  # 여기를 실제 app.invoke 호출로 바꾸세요
    output_json = json.dumps(output)
    
    # 디렉토리 생성
    directory_path = os.path.join(output_path, country, category)
    os.makedirs(directory_path, exist_ok=True)
    
    # augmented_prompt 저장
    output_data = json.loads(output_json)
    output_prompts = output_data["augmented_prompt"]

    
    file_name = f"{country}_{category}_{cultural_noun}.txt"
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, 'a', encoding='utf-8') as file:
        ## 나중에 ast 문제 해결되면 주석으로 변경
        # file.write(output_prompts + '\n')
        for prompt in output_prompts:
            try:
                file.write(prompt + "\n")
            except:
                print(f"ERROR : {cultural_noun} // {prompt}")
                continue
    
    return output_data["score"], output_data['score_history'], output_data["refine_recur_counter"]
                
# 경로 설정
cultural_keywords_path = 'cultural_keyword/keywords_0810/target'
prompts_directory = 'sentence_classification/prompts'
output_path = 'results/test'

# 카테고리 목록
categories = [
    # "architecture",
    # "city(landmark)",
    "clothing",
    # "dance music",
    # "food and drink",
    # "religion and festival",
    # "utensils and tools",
    # "visual arts" 
]

category_file_name = [  # 수정된 부분
    # "architecture.txt",
    # "city_landmark.txt",
    "clothing.txt",
    # "dance_music.txt",
    # "food_drink.txt",
    # "religion_festival.txt",
    # "utensil_tool.txt",
    # "visual_arts.txt"
]

# 모든 나라의 데이터를 읽어오기
all_country_data = read_all_json_files(cultural_keywords_path)

# prompts 디렉토리에서 카테고리별 문장 가져오기 - 수정된 부분
category_prompts = get_prompts_from_directory(prompts_directory, categories, category_file_name)

# 모든 나라의 데이터로부터 문장 생성
all_sentences = generate_sentences_from_all_data(all_country_data, category_prompts)


# 매 반복마다의 점수를 저장할 리스트 초기화

results = []

# 13개씩 묶어서 app.invoke 호출
batch_size = 13
for sentence in tqdm(all_sentences, desc="Processing", unit="item"):
    
    cultural_noun = sentence["cultural_noun"]
    category = sentence["category"]
    prompts = sentence["prompts"]

    country = sentence["country"]
    
    score, history, counter = app_invoke(cultural_noun, category, prompts, country)

    result_entry= {
        "score": score,
        "score_history": history,
        "refine_counter": counter,
        "metadata": {
            'country': country,
            'category': category,
            'cultural_noun': cultural_noun
        }
    }
    results.append(result_entry)
    
    with open(output_path+'/score_with_counter.jsonl', 'a') as file:
        file.write(json.dumps(result_entry) + '\n')
    
    
    for i in range(0, len(prompts), batch_size):
        if (len(prompts) - i) <= batch_size:
            # 마지막 루프 처리
            batch_prompts = prompts[i:]
            current_batch = len(prompts) - i
        else:
            batch_prompts = prompts[i:i + batch_size]
            current_batch = batch_size
        
        

# 생성된 문장 수 확인
print(f"\nTotal inputs generated: {len(all_sentences)}")


# In[ ]:


# 실제 실험 구성
# 데이터 불러와서 랭그래프 실행 후 다시 저장

# if False:
#     ## type 1 
#     inputs = {
#         "cultural_noun": "Beijing",
#         "prompt": "The city is good.",
#         }

#     # output = app.invoke(inputs)
#     # print(output)

#     for s in app.stream(inputs, config={"configurable": {"thread_id": 1111}}):
#         print(list(s.values())[0])
#         print("----")
    
#     import os
#     import json

#     data_path = "cultural_keyword/keywords"
#     data_list = os.listdir(data_path)
#     output_path = "outputs"

#     output_data = {}

#     for data in data_list:
#         print('현재 처리 중인 데이터:',data)
#         with open(data_path + '/' + data, 'r', encoding='utf-8') as f:
#             json_data = json.load(f)

#         for k, v in json_data.items():
#             outputs = []
#             for keyword in v: 
#                 inputs = {"cultural_noun": keyword}
#                 output = app.invoke(inputs)
                
#                 keyword_output = {"keyword": keyword, "caption": output}
#                 outputs.append(keyword_output)
#             output_data[k] = outputs

#         with open(output_path + '/' + data, 'w', encoding='utf-8') as f:
#             json.dump(output_data, f, indent=4)

