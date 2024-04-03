import re
import json
import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
from ollama import Client
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
from threading import Thread
import torch


client = Client(host='172.16.30.181:30620')



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

def extract_user_mysql(target_user):
    from sqlalchemy import create_engine
    import pandas as pd
    print(target_user)
    con = create_engine(f'mysql+pymysql://root:root@172.16.30.212/')
    uids_sql = f"select * from zr.userdatas where 用户昵称='{target_user}'"
    print(uids_sql)
    uid_data = pd.read_sql(sql=uids_sql, con=con.connect())
    user_list = uid_data.to_dict(orient='records')
    print(user_list)

    uids_sql1 = f"select 性别,是否认证,认证理由,简介,设备,粉丝数,创建时间,用户等级,关注数,互关数 from zr.userdatas where 用户昵称='{target_user}'"
    uid_data1 = pd.read_sql(sql=uids_sql1, con=con.connect())
    user_attr = uid_data1.to_dict(orient='records')
    
    return user_list,user_attr



def detailed_report_user(model_selection, temperature, myprompt):


    print(myprompt)
    print('------------------------------')
    print('大模型开始工作')
    print('------------------------------')
    print(model_selection)
    if model_selection != 'ChatGLM3-6B':
        partial_message = ""
        for response in client.generate(model=model_selection, prompt=myprompt, options={'temperature': temperature, },
                                        stream=True, ):
            # print(response)
            content = response['response']
            # print(content, end='', flush=True)

            partial_message += content
            # print("partial_message================",partial_message)
            # print("[[None, partial_message]================",[[None, partial_message]])
            yield [partial_message]

    else:
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                stop_ids = [29, 0]
                for stop_id in stop_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        history_transformer_format = [[myprompt, ""]]
        stop = StopOnTokens()

        messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
                            for item in history_transformer_format])
        tokenizer = AutoTokenizer.from_pretrained("/home/LLM/chatglm3-6b/", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("/home/LLM/chatglm3-6b/",
                                                    torch_dtype=torch.float16, device_map="auto",
                                                    trust_remote_code=True)
        model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=temperature + 0.001,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != '<':
                partial_message += new_token
                return [[None, partial_message]]
# kg="""
# ["{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}","{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}","{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}"]
# """
# result = extract_and_concatenate_values(kg, 'id')
# print("-----")
# print(result)

# import re

# kg = """
# ["{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}","{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}","{'id': '4845136662235585', 'pmid': '4845136548202356', '平台': '微博', 'publisher_id': '7276797259', '用户': '降雨概率', 'source_publisher_id': '7060541217', '原始发布者': '貓耳海豚', '时间': '2022-12-10 14:34:21', '内容': '//@貓耳海豚://@道歉要露出肚皮喲:被转量微博 ###看的我直发抖 原来冰冷的”新闻”字眼只会忽略事情的严重性 #警方回应鹤壁16岁女生遭霸凌# \\u200b\\u200b\\u200b', '事件': '鹤壁女孩', '情绪': '负面', '地址': '广东', 'wtype': '2', '具体情感关键词': '其他', '阅读量': 0, '点赞': 1, 'comment_num': 0, '被转量': 0, 'day': None, 'insert_time': '20230823'}"]
# """


