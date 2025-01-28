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
