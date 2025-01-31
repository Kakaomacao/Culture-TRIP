import os
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from utils.data_loader import cultural_noun_load
from iterative_refinement.iterative_refinement import retrieve_info, refiner, evaluator, feedbacker, keyword_prompt, llm_augment_prompt, check_feedback

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

def setup_workflow():
    """Set up the workflow for the Culture-TRIP"""
    workflow = StateGraph(GraphState)

    workflow.add_node("Load culture nouns", cultural_noun_load) 
    workflow.add_node("Retrieve information", retrieve_info) 
    workflow.add_node("Refine information", refiner)
    workflow.add_node("Evaluate information", evaluator)
    workflow.add_node("Feedback", feedbacker)
    workflow.add_node("Add culture nouns", keyword_prompt)  
    workflow.add_node("Augent prompts", llm_augment_prompt)  

    workflow.add_edge("Load cultural nouns", "Retrieve information")
    workflow.add_edge("Retrieve information", "Refine information")
    workflow.add_edge("Refine information", "Evaluate information")
    workflow.add_conditional_edges(
        "Evaluate information",
        check_feedback,
        {
            "sufficient": "Add prompt keywords",
            "insufficient": "Feedback"
        }
    ) 
    workflow.add_edge("Feedback", "Refine information")
    workflow.add_edge("Add prompt keywords", "Augment prompts") 
    workflow.add_edge("Augment prompts", END)

    workflow.set_entry_point("Load cultural nouns")
    return workflow

def app_invoke(cultural_noun, category, prompts, country, app):
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
