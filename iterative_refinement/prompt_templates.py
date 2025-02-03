from langchain_core.prompts import PromptTemplate

# Define various prompt templates for writing, evaluating, refining, and augmenting prompts
# Template1: Generate detailed sentences for culture nouns
refine_prompt_template = """### Instruction:
Please revise BASE PROMPT by refferring to the INFORMATION as noted in the FEEDBACK.
I will present BASE PROMPT to someone unfamiliar with the CULTURE NOUN, so they can draw a picture of the CULTURE NOUN just by reading the base PROMPT.

There may be incorrect information in the INFORMATION, so be cautious and ensure it pertains to the CULTURE NOUN before using it.

If a BASE PROMPT cannot accommodate the CULTURE NOUN, allow for slight modifications to ensure all sentences are covered. When adding additional information to a single sentence to provide sufficient detail, expand the original sentence into 3 sentences so that the result is approximately 300 characters long.

### TODO:
CULTURE NOUN:
{culture_noun}

INFORMATION:
{information}

BASE PROMPT:
{prompt}

BEFORE REFINED PROMPT:
{refined_prompt}

REFINE FEEDBACK:
{feedback}

ANSWER:
"""
refine_prompt = PromptTemplate.from_template(refine_prompt_template)


# Template2: Evaluate prompts based on five criteria
scoring_prompt_template = """### Instruction:
Please evalutate base PROMPT with 5 criteria (Clarity, Detail, Context, Purpose, Comparable object).
- Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the CULTURE NOUN.
- Visual detail: Whether the prompt provides a sufficient amount of visual information, such as colors, shapes, etc.
- Background: Whether the historical or temporal background information provided in the prompt is appropriate.
- Purpose: Whether the description of the intended use or the users of the subject in the prompt is appropriate.
- Comparable object: How well the prompt compares to existing well-known or popular examples.
Each criterion cannot exceed a score of 10. Please provide each criterion's score and the total score.
Answer in the following foramt and write only the score, not the description.

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
CULTURE NOUN:
{culture_noun}

REFINED PROMPT:
{refined_prompt}

ANSWER:
"""
scoring_prompt = PromptTemplate.from_template(scoring_prompt_template)


# Template3: Provide feedback to improve prompt scores
feedback_prompt_template = """### Instruction:
Review the items of SCORE and provide feedback on how to improve each item's score, specifically focusing on the modification of REFINED PROMPT about CULTURE NOUN.

### TODO:
CULTURE NOUN:
{culture_noun}

REFINED PROMPT:
{refined_prompt}

SCORE:
{score}

ANSWER:
"""

feedback_prompt = PromptTemplate.from_template(feedback_prompt_template)