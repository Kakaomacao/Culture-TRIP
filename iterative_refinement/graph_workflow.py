from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from iterative_refinement.iterative_refinement import GraphState, culture_noun_load, retrieve_info, refine, scoring, feedback, check_score

def setup_workflow():
    """Set up the workflow for the Culture-TRIP"""
    workflow = StateGraph(GraphState)

    workflow.add_node("Load culture nouns", culture_noun_load) 
    workflow.add_node("Retrieve information", retrieve_info) 
    workflow.add_node("Refine prompt", refine)
    workflow.add_node("Evaluate prompt", scoring)
    workflow.add_node("Feedback", feedback)

    workflow.add_edge("Load culture nouns", "Retrieve information")
    workflow.add_edge("Retrieve information", "Refine prompt")
    workflow.add_edge("Refine prompt", "Evaluate prompt")
    workflow.add_conditional_edges(
        "Evaluate prompt",
        check_score,
        {
            "sufficient": END,
            "insufficient": "Feedback"
        }
    ) 
    workflow.add_edge("Feedback", "Refine prompt")

    workflow.set_entry_point("Load culture nouns")
    return workflow

def culture_trip(culture_noun, prompt, is_intermediate_result_show):
    # Setup the langchain workflow
    workflow = setup_workflow()
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    inputs = {
        "culture_noun": culture_noun,
        "prompt": prompt,
        "is_intermediate_result_show": is_intermediate_result_show
    }
    output = app.invoke(inputs, config={"configurable": {"thread_id": 1111}, "recursion_limit": 100})

    # output_json = json.dumps(output)
    # output_data = json.loads(output_json)
    # output_prompts = output_data["refined_prompt"]

    return output["refined_prompt"]