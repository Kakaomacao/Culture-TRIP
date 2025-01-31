import json
from tqdm import tqdm
from langgraph.checkpoint.memory import MemorySaver
from utils.data_loader import read_all_json_files, get_prompts_from_directory, generate_sentences_from_all_data
from utils.graph_workflow import setup_workflow, app_invoke, batch_size

# Load environment variables (API keys)
from dotenv import load_dotenv
load_dotenv()

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

# TODO: Get path by user input(passing arguments)
# Define paths for input and output
cultural_keywords_path = 'cultural_keyword/keywords_0810/target'
prompts_directory = 'sentence_classification/prompts'
output_path = 'results/test'

# Load all data from the input directory
all_country_data = read_all_json_files(cultural_keywords_path)
category_prompts = get_prompts_from_directory(prompts_directory, categories, category_file_name)
all_sentences = generate_sentences_from_all_data(all_country_data, category_prompts)

# Setup the workflow
workflow = setup_workflow()
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Process all generated sentences through the workflow
results = []
for sentence in tqdm(all_sentences, desc="Processing", unit="item"):
    
    cultural_noun = sentence["cultural_noun"]
    category = sentence["category"]
    prompts = sentence["prompts"]

    country = sentence["country"]
    
    score, history, counter = app_invoke(cultural_noun, category, prompts, country, app)

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
