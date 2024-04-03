import re
import json

def count_id_occurrences(text, target_id):
    occurrences = text.count(target_id)
    return occurrences



def concatenate_content(text):
    pattern = r'"":\s*"([^"]*)"'
    contents = re.findall(pattern, text)
    concatenated_content = " ".join(contents)
    return concatenated_content



def extract_and_concatenate_values1(text, key):
    if text.startswith("['") and text.endswith("']"):
        text = text[2:-2]
    print(text)
    data = json.loads(text)
    values = []
    for item in data:
        value = item.get(key)
        if value:
            values.append(value)
    concatenated_values = " ".join(values)
    return concatenated_values

def extract_and_concatenate_values2(text, key):
    # pattern = r'{.*?"' + key + r'"\s*:\s*"(.*?)".*?}'
    pattern = r'"' + key + r'"\s*:\s*"(.*?)"'
   
    matches = re.findall(pattern, text, re.DOTALL)
  
    return matches



def extract_user_data(json_file, target_user):
 
    with open(json_file,'r') as f:
        data = json.load(f)
  
    concatenated_data = []

    for item in data['datas']:
        if item['用户昵称'] == target_user:
            attr_data=list(item.items())[3:15]
            concatenated_data.append(str(item))
    
    return concatenated_data,attr_data

def extract_quotes_content(string):
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, string)
    return matches


def extract_and_concatenate_values(text, key):
    pattern = r"'" + key + r"'\s*:\s*'([^']*)'"
    matches = re.findall(pattern, text)
    return matches



