import os
import json

def read_json(file_path):
    """JSON 파일 읽기"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def read_all_json_files(folder_path):
    """폴더 내 모든 JSON 파일 읽기"""
    all_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            country_name = file_name.split('.')[0]
            all_data[country_name] = read_json(file_path)
    return all_data

def save_to_file(data, file_path):
    """데이터를 JSONL 형식으로 저장"""
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + '\n')

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
