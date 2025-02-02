from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from iterative_refinement.iterative_refinement import GraphState, cultural_noun_load, retrieve_info, refiner, evaluator, feedbacker, keyword_prompt, llm_augment_prompt, check_feedback

def setup_workflow():
    """Set up the workflow for the Culture-TRIP"""
    workflow = StateGraph(GraphState)

    workflow.add_node("Load culture nouns", cultural_noun_load) 
    workflow.add_node("Retrieve information", retrieve_info) 
    workflow.add_node("Refine prompt", refiner)
    workflow.add_node("Evaluate prompt", evaluator)
    workflow.add_node("Feedback", feedbacker)

    workflow.add_edge("Load culture nouns", "Retrieve information")
    workflow.add_edge("Retrieve information", "Refine prompt")
    workflow.add_edge("Refine prompt", "Evaluate prompt")
    workflow.add_conditional_edges(
        "Evaluate prompt",
        check_feedback,
        {
            "sufficient": END,
            "insufficient": "Feedback"
        }
    ) 

    workflow.set_entry_point("Load culture nouns")
    return workflow

def culture_trip(cultural_noun, prompts):
    # Setup the langchain workflow
    workflow = setup_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    inputs = {
        "cultural_noun": cultural_noun,
        "prompts": prompts
    }
    output = app.invoke(inputs, config={"configurable": {"thread_id": 1111}, "recursion_limit": 100})

    # output_json = json.dumps(output)
    # output_data = json.loads(output_json)
    # output_prompts = output_data["augmented_prompt"]

    return output["augmented_prompt"]