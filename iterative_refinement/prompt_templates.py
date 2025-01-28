from langchain_core.prompts import PromptTemplate

# Define various prompt templates for writing, evaluating, refining, and augmenting prompts
# Template1: Generate detailed sentences for cultural nouns
template1 = """### Instruction:
I want you to write 2 ~ 3 detailed sentences with INFORMATION.
Please revise the CONTEXT content by referring to the INFORMATION as noted in the FEEDBACK.
There may be incorrect information in the INFORMATION, so be cautious and ensure it pertains to the CULTURAL NOUN before using it.

### TODO:
CULTURAL NOUN:
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


# Template2: Evaluate prompts based on five criteria
template2 = """### Instruction:
Please evalutate CONTEXT with 5 criteria (Clarity, Detail, Context, Purpose, Comparable object).
Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the CULTURAL NOUN.
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
CULTURAL NOUN:
{cultural_noun}

CONTEXT:
{context}

Answer:
"""
prompt2 = PromptTemplate.from_template(template2)


# Template3: Provide feedback to improve prompt scores
template3 = """### Instruction:
Review the items of SCORE and provide feedback on how to improve each item's score, specifically focusing on the modification of CONTEXT about CULTURAL NOUN.

### TODO:
CULTURAL NOUN:
{cultural_noun}

CONTEXT:
{context}

SCORE:
{score}

Answer:
"""

prompt3 = PromptTemplate.from_template(template3)


# Template4: Add culture keywords to base prompts
template4 = """### Instruction:
The CULTURAL NOUN and BASE PROMPTS are related to the CATEGORY. 
In each of the Base Prompts, there is a mask coverd with {{ }} and replace them with CULTURAL NOUN including {{}}.

Please answer in the FORMAT below. I want to have an ANSWER in format of list.
ANSWER FORMAT:
["sentence", "sentence", ...]

### TODO:
CULTURAL NOUN:
{cultural_noun}

Base Prompts:
{prompts}

Answer:
"""

prompt4 = PromptTemplate.from_template(template4)


# Template5 : Augmenting the prompt
template5 = """### Instruction:
Use all the information from the CONTEXT to add adtional information to each sentence of the Base prompts. Explain it so well that someone who doesn't know the CULTURAL NOUN can draw a picture of the CULTURAL NOUN just by reading the Base prompts. 

Refer to the following criteria:
Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the CULTURAL NOUN.
Visual detail: Whether the prompt provides a sufficient amount of visual information, such as colors, shapes, etc.
Background: Whether the historical or temporal background information provided in the prompt is appropriate.
Purpose: Whether the description of the intended use or the users of the subject in the prompt is appropriate.
Comparable object: How well the prompt compares to existing well-known or popular examples.

If a Base prompt cannot accommodate the CULTURAL NOUN, allow for slight modifications to ensure all sentences are covered. 
When adding additional information to a single sentence to provide sufficient detail, expand the original 1 sentence into 3 sentences so that the final result is approximately 300 characters long.

Please answer in the FORMAT below. I want to have an ANSWER in format of list.
ANSWER FORMAT:
["3 sentences", "3 sentences", ...]

### TODO:
CULTURAL NOUN:
{cultural_noun}

CONTEXT:
{context}

Base Prompts:
{prompts}

Answer:
"""

prompt5 = PromptTemplate.from_template(template5)
