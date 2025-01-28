#!/usr/bin/env python
# coding: utf-8

import os
import ast
import json
from time import sleep
import re
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

from typing import TypedDict

# Define a structure for storing graph state data
class GraphState(TypedDict):
    cultural_noun: str
    category: str
    prompts: list[str]
    wikisearch_context: str
    googlesearch_context: str
    context: str
    score_history: dict
    score: dict
    feedback: str
    augmented_prompt: list[str]
    refine_recur_counter: int

# Import necessary LangChain modules

from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from tqdm import tqdm

# Initialize LLM with a specific model
llm = ChatOllama(model="llama3:70b")

from utils.prompt_templates import prompt1, prompt2, prompt3, prompt4, prompt5
llm_chain1 = prompt1 | llm | StrOutputParser()
llm_chain2 = prompt2 | llm | StrOutputParser()
llm_chain3 = prompt3 | llm | StrOutputParser()
llm_chain4 = prompt4 | llm | StrOutputParser()
llm_chain5 = prompt5 | llm | StrOutputParser()

from rag.utils import format_searched_web, format_searched_docs
from rag.customWiki import CustomWikipediaAPI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper

GRAPH_SLEEP_TIME = 0  # Delay for each graph operation
GRPAH_RESULT_SHOW = True  # Toggle for displaying intermediate results

# Initialize Wikipedia and Google search tools
wsearch = WikipediaQueryRun(api_wrapper=CustomWikipediaAPI())
gsearch = GoogleSearchAPIWrapper()


# Initialize the graph state with default values
def cultural_noun_load(state: GraphState) -> GraphState:
    state['augmented_prompt'] = ''
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### GRAPH START ###")
        print("CULTURAL NOUN:", state['cultural_noun'])
    
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

# Retrieve information via Wikipedia and Google searches
def retrieve_info(state: GraphState) -> GraphState:
    wikisearch_context = wsearch.run(state["cultural_noun"])
    googlesearch_context = gsearch.run("What is " + state["cultural_noun"])
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### RETRIEVE INFO ###\n<wiki>"+wikisearch_context+'\n<google>'+googlesearch_context)

    sleep(GRAPH_SLEEP_TIME)
    return GraphState(wikisearch_context=wikisearch_context, googlesearch_context=googlesearch_context)

# Refine the retrieved context based on feedback
def refiner(state: GraphState) -> GraphState:
    information = state["wikisearch_context"] + '\n' + state["googlesearch_context"]
    feedback = state["feedback"]
    
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

# Evaluate the refined context and score it
def evaluator(state: GraphState) -> GraphState:    
    response = llm_chain2.invoke({
        "cultural_noun":state["cultural_noun"],
        "category":state["category"],
        "context":state["context"],
    })

    print("\n\n### EVALUATOR ###") if GRPAH_RESULT_SHOW else None

    json_match = re.search(r'\{.*?\}', response, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        score = json.loads(json_str)
        
        score_history = state['score_history']        
        for key in score_history.keys():
            score_history[key].append(score[key])
            
        print(f'score : {score}') if GRPAH_RESULT_SHOW else None
    else:
        print("JSON data not found") 

    sleep(GRAPH_SLEEP_TIME)   
    
    return GraphState(score=score, score_history=score_history)

# Provide feedback to improve the context based on scores
def feedbacker(state: GraphState) -> GraphState:
    context = state["context"]
    keyword = state["cultural_noun"]
    score = state["score"]
    counter = state["refine_recur_counter"] + 1
    
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

# Replace placeholders in prompts with cultural nouns
def keyword_prompt(state: GraphState) -> GraphState:
    prompts = state["prompts"]
    cultural_noun = state["cultural_noun"]
    formatted_prompts = []
    
    for prompt in prompts:
        matches = re.findall(r'\{(.*?)\}', prompt)
        
        for match in matches:
            prompt = prompt.replace(f'{{{match}}}', cultural_noun)

        formatted_prompts.append(prompt)
    
    if GRPAH_RESULT_SHOW:
        print("\n\n### ADD KEYWORD ###")
        print(formatted_prompts)    
    
    return GraphState(prompts=formatted_prompts)

# Check if the feedback process has sufficiently refined the context
def check_feedback(state: GraphState) -> GraphState:
    feedback_score = state['score']
    total = feedback_score["Total_score"]
    counter = state["refine_recur_counter"]
    THRESH = 40

    if total >= THRESH or counter >= 5:
        return "sufficient"
    else:
        return "insufficient"

# Augment prompts with detailed descriptions
def llm_augment_prompt(state: GraphState) -> GraphState:    
    pass
    print("\n\n### AUGMENTATION ###") if GRPAH_RESULT_SHOW else None

    prompts = state["prompts"]
    total_aug = []
    
    # Process prompts in batches for augmentation
    for i in range(0, len(prompts), batch_size):
        aug_recur_counter = 0
        aug_fail = True
        
        if (len(prompts) - i) <= batch_size:
            batch_prompts = prompts[i:]
            current_batch = len(prompts) - i
        else:
            batch_prompts = prompts[i:i + batch_size]
            current_batch = batch_size
        
        # Retry augmentation up to 5 times in case of failure
        while aug_recur_counter < 5 and aug_fail:
            aug_recur_counter += 1
            
            response = llm_chain5.invoke({
                "cultural_noun":state["cultural_noun"], 
                "context":state["context"],
                "prompts":batch_prompts
            })
            
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            
            if list_match:
                try:
                    list_str = list_match.group(0)
                    augmented_prompt = json.loads(list_str)
                    
                    # Ensure the batch was processed successfully
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
                
                # Log errors to a file for debugging
                with open('error.txt', 'a') as error_file:
                    error_file.write('------------------------\n')
                    error_file.write(f'response: {response}\n')
                    error_file.write(f'list_match: {list_match}\n')
                    error_file.write('------------------------\n')
        
        total_aug.extend(augmented_prompt)
        
    state["wikisearch_context"] = ''
    state["googlesearch_context"] = ''
    state["context"] = ''

    return GraphState(augmented_prompt=total_aug)


# Workflow setup: Define the sequence of operations in the graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from utils.graph_workflow import setup_workflow

workflow = setup_workflow()

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)


# Utility functions for loading and processing JSON files
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_all_json_files(folder_path):
    all_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            country_name = file_name.split('.')[0]
            all_data[country_name] = read_json(file_path)
    return all_data

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

# Generate sentences by combining all country and category data
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

# Invoke the workflow for each cultural noun and save results
def app_invoke(cultural_noun, category, prompts, country):
    global output_path
    
    inputs = {
        "cultural_noun": cultural_noun,
        "category": category,
        "prompts": prompts
    }
    print(f'{category} : {cultural_noun} of {country}')
    output = app.invoke(inputs, config={"configurable": {"thread_id": 1111}, "recursion_limit": 100})
    output_json = json.dumps(output)
    
    # Check directory
    directory_path = os.path.join(output_path, country, category)
    os.makedirs(directory_path, exist_ok=True)
    
    output_data = json.loads(output_json)
    output_prompts = output_data["augmented_prompt"]

    file_name = f"{country}_{category}_{cultural_noun}.txt"
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, 'a', encoding='utf-8') as file:
        for prompt in output_prompts:
            try:
                file.write(prompt + "\n")
            except:
                print(f"ERROR : {cultural_noun} // {prompt}")
                continue

    return output_data["score"], output_data['score_history'], output_data["refine_recur_counter"]

# Set paths for inputs and outputs
cultural_keywords_path = 'cultural_keyword/keywords_0810/target'
prompts_directory = 'sentence_classification/prompts'
output_path = 'results/test'

categories = [
    "architecture",
    "city(landmark)",
    "clothing",
    "dance music",
    "food and drink",
    "religion and festival",
    "utensils and tools",
    "visual arts" 
]

category_file_name = [
    "architecture.txt",
    "city_landmark.txt",
    "clothing.txt",
    "dance_music.txt",
    "food_drink.txt",
    "religion_festival.txt",
    "utensil_tool.txt",
    "visual_arts.txt"
]

all_country_data = read_all_json_files(cultural_keywords_path)

category_prompts = get_prompts_from_directory(prompts_directory, categories, category_file_name)

all_sentences = generate_sentences_from_all_data(all_country_data, category_prompts)

# Process all generated sentences through the workflow
results = []
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
            batch_prompts = prompts[i:]
            current_batch = len(prompts) - i
        else:
            batch_prompts = prompts[i:i + batch_size]
            current_batch = batch_size

print(f"\nTotal inputs generated: {len(all_sentences)}")
