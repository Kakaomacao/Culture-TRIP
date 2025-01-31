from iterative_refinement.customWiki import CustomWikipediaAPI
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from iterative_refinement.prompt_templates import prompt1, prompt2, prompt3, prompt4, prompt5

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

# ---------------------------------------------------------
# Search functions
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
# LLM-related functions
# ---------------------------------------------------------

def refine_context(cultural_noun, category, information, context, feedback):
    """
    Refine the given context based on the retrieved information and feedback.

    Args:
        cultural_noun (str): The cultural noun being refined.
        category (str): The category of the cultural noun.
        information (str): The information retrieved from searches.
        context (str): The current context to refine.
        feedback (str): Feedback for improving the context.

    Returns:
        str: The refined context.
    """
    response = llm_chain1.invoke({
        "cultural_noun": cultural_noun,
        "category": category,
        "information": information,
        "context": context,
        "feedback": feedback
    })
    return response

def evaluate_context(cultural_noun, category, context):
    """
    Evaluate the given context using predefined criteria.

    Args:
        cultural_noun (str): The cultural noun being evaluated.
        category (str): The category of the cultural noun.
        context (str): The context to evaluate.

    Returns:
        dict: A dictionary containing the evaluation scores.
    """
    response = llm_chain2.invoke({
        "cultural_noun": cultural_noun,
        "category": category,
        "context": context
    })
    return parse_json_response(response)

def parse_json_response(response):
    """
    Parse a JSON response from the LLM.

    Args:
        response (str): The response string from the LLM.

    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    import re
    import json

    json_match = re.search(r'\{.*?\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    else:
        return {}

# ---------------------------------------------------------
# Feedback functions
# ---------------------------------------------------------

def generate_feedback(cultural_noun, category, context, score):
    """
    Generate feedback for improving the context based on evaluation scores.

    Args:
        cultural_noun (str): The cultural noun being evaluated.
        category (str): The category of the cultural noun.
        context (str): The context that was evaluated.
        score (dict): The evaluation scores.

    Returns:
        str: Feedback for improving the context.
    """
    response = llm_chain3.invoke({
        "cultural_noun": cultural_noun,
        "category": category,
        "context": context,
        "score": score
    })
    return response

def check_feedback(score, refine_counter, threshold=40, max_attempts=5):
    """
    Check whether the feedback process has achieved sufficient refinement.

    Args:
        score (dict): The evaluation scores.
        refine_counter (int): The number of refinement attempts made.
        threshold (int): The minimum total score to consider the context sufficient.
        max_attempts (int): The maximum number of refinement attempts allowed.

    Returns:
        str: 'sufficient' if refinement is complete, 'insufficient' otherwise.
    """
    total = score.get("Total_score", 0)
    return "sufficient" if total >= threshold or refine_counter >= max_attempts else "insufficient"
