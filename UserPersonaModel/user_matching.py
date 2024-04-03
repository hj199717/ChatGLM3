# -*-coding:utf-8 -*-
'''
File       : gr001.py
Time       : 2024/3/18 15:00
Author     : He Jia
version    : 1.0.0
Description: 
'''

import gradio as gr
import pandas as pd
from sqlalchemy import create_engine
from ollama import Client
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
from threading import Thread
import torch

# tokenizer = AutoTokenizer.from_pretrained("/home/LLM/chatglm3-6b/", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("/home/LLM/chatglm3-6b/",
#                                              torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

client = Client(host='172.16.30.181:30620')


# print(gr.__version__)
def illustrate_pattern(pattern):
    if pattern == "跨网域匹配":
        return "跨网域匹配"
    elif pattern == "跨平台匹配":
        return "跨平台匹配"
    elif pattern == "同平台匹配":
        return "同平台匹配"
    else:
        return ""


def find_user_pd(user1, plat):
    if plat == '微博':
        table_name = 'suzhou1st'  # 处理原表名称
        database = 'weibo'
        host = '172.16.30.216'
        port = 3306
        user = 'root'
        password = 'root'

        mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
        con = create_engine(f'{mysql_ip}{database}')
        df = pd.read_sql(f"select * from {table_name} where screen_name='{user1}'", con)
        # df = df.iloc[0,:]
        df = df[['screen_name', 'uid', 'gender', 'location', 'mbrank', 'post_count',
                 'followers_count', 'friends_count', 'description',
                 'verified_reason']]
        df.fillna('无', inplace=True)
        gender_dic = {'m': '男', 'f': '女'}
        df['gender'] = df.apply(lambda x: gender_dic.get(x.gender, '未知'), axis=1)
        df.columns = ['昵称', 'uid', '性别', '位置', '等级', '发帖数', '粉丝数', '关注数', '个人描述', '认证原因']
    else:
        table_name = 'ods_internet_toutiao_user'
        database = 'toutiao'
        host = '172.16.30.216'
        port = 3306
        user = 'root'
        password = 'root'

        mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
        con = create_engine(f'{mysql_ip}{database}')
        df = pd.read_sql(f"select * from {table_name} where name='{user1}'", con)
        # df = df.iloc[0,:]
        df = df[['name', 'uid', 'location', 'digg_count',
                 'follower_count', 'focus_on_count', 'desc',
                 'auth']]
        df['auth'] = df['auth'].apply(lambda x: x.split('：')[1].strip() if isinstance(x, str) else x)
        df['location'] = df['location'].apply(lambda x: x.split('：')[1].strip() if isinstance(x, str) else x)
        df.fillna('无', inplace=True)
        df.columns = ['昵称', 'uid', '位置', '获赞数', '粉丝数', '关注数', '个人描述', '认证原因']

    return df


# def find_2_user_detail(user1, user2):
#     table_name = 'suzhou1st'  # 处理原表名称
#     host = '172.16.30.216'
#     port = 3306
#     user = 'root'
#     password = 'root'
#     database = 'weibo'
#     mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
#     con = create_engine(f'{mysql_ip}{database}')
#     df1 = pd.read_sql(f"select * from {table_name} where screen_name='{user1}'", con)
#     print(df1)
#     print('---------------------------------')
#     df2 = pd.read_sql(f"select * from {table_name} where screen_name='{user2}'", con)
#     print(df2)
#     print('-------------------------------------------')
#     print(df2.columns)
#
#     def transfer_df(df: pd.DataFrame):
#         df = df[['screen_name', 'uid', 'gender', 'location', 'mbrank', 'post_count',
#                  'followers_count', 'friends_count', 'description',
#                  'verified_reason']]
#         df.fillna('无', inplace=True)
#         gender_dic = {'m': '男', 'f': '女'}
#         df['gender'] = df.apply(lambda x: gender_dic.get(x.gender, '未知'), axis=1)
#         user_info = df.iloc[0, :].tolist()
#         text = f'用户昵称为“{user_info[0]}”，该用户的uid为{user_info[1]}，该用户的性别为{user_info[2]}，该用户的地理位置为{user_info[3].replace(" ", "")}，\
# 该用户的等级为{user_info[4]}级，该用户的发帖总数为{user_info[5]}，该用户的粉丝数为{user_info[6]}，该用户的关注数为{user_info[7]}，\
# 该用户的用户个人描述为“{user_info[8]}”，该用户的认证原因为“{user_info[9]}”。'
#         return text
#
#     text1 = transfer_df(df1)
#     text2 = transfer_df(df2)
#     return text1, text2

def find_2_user_detail(user1, user2, plat1, plat2):
    df1 = find_user_pd(user1, plat1)
    df2 = find_user_pd(user2, plat2)

    def transfer_df(df: pd.DataFrame):
        first_row = df.iloc[0]
        text = "\n".join([f"{column_name}: {value}" for column_name, value in first_row.items()])
        return text

    text1 = transfer_df(df1)
    text2 = transfer_df(df2)
    return text1, text2


def find_match_user_list(user_name, plat2):
    if plat2 == '微博':
        # 此处调整MySQL相关参数
        table_name = 'net_match'  # 处理原表名称
        database = 'net_match'
    else:
        table_name = 'net_pair_raw_table_weibo2toutiao'  # 处理原表名称
        database = 'hj_web'
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
    con = create_engine(f'{mysql_ip}{database}')

    # user_name='天翼帐号11004444710'

    df = pd.read_sql(f"select * from {table_name} where name='{user_name}'", con)
    # print(len(df))
    # print(df)
    l = len(df)

    if l == 0:
        return '未匹配到用户'
    else:
        data = df.iloc[0, :]
        print(data)

        match_lst = []
        for i in range(5):
            i_name = data[f'{i + 1}name']
            if not pd.isna(i_name):
                match_lst.append(i_name)
        print(match_lst)
        table_colomn_2 = gr.Dropdown(value=match_lst[0], choices=match_lst)
        return table_colomn_2
        # match_name.choices=match_lst
        # match_user = match_lst[0]
        # return match_user
        # return match_lst[0]


def detailed_report(model_selection, temperature, user1, user2, plat1, plat2):
    # yield [[None,None]]
    text1, text2 = find_2_user_detail(user1, user2, plat1, plat2)
    # myprompt = f'## 现在已经确认以下两个微博用户隶属同一自然人，两个微博用户的详细资料如下所示，请分析两个用户的相似性，\
    # 并指出两个微博用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。\
    # \n第一个用户的资料为：{text1}\n第二个用户的资料为：{text2}'
    myprompt = f'## 现在已经确认以下两个互联网用户隶属同一自然人，两个用户的详细资料如下所示，请分析两个用户的相似性，\
并指出两个用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。\n\
## 第一个用户为{plat1}平台的用户，其资料为：\n{text1}\n## 第二个用户为{plat2}平台的用户，其资料为：\n{text2}'
    print(myprompt)
    print('------------------------------')
    print('大模型开始工作')
    print('------------------------------')
    print(model_selection)
    print(list(model_selection))
    if model_selection != 'ChatGLM3-6B':
        partial_message = ""
        for response in client.generate(model=model_selection, prompt=myprompt, options={'temperature': temperature, },
                                        stream=True, ):

            print(response)
            content = response['response']
            print(content, end='', flush=True)
            partial_message += content
            yield [[None, partial_message]]

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
                yield [[None, partial_message]]

# def detailed_report(model_selection, temperature, user1, user2, plat1, plat2):
#     # yield [[None,None]]
#     text1, text2 = find_2_user_detail(user1, user2, plat1, plat2)
#     # myprompt = f'## 现在已经确认以下两个微博用户隶属同一自然人，两个微博用户的详细资料如下所示，请分析两个用户的相似性，\
#     # 并指出两个微博用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。\
#     # \n第一个用户的资料为：{text1}\n第二个用户的资料为：{text2}'
#     myprompt = f'## 现在已经确认以下两个互联网用户隶属同一自然人，两个用户的详细资料如下所示，请分析两个用户的相似性，\
# 并指出两个用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。\n\
# ## 第一个用户为{plat1}平台的用户，其资料为：\n{text1}\n## 第二个用户为{plat2}平台的用户，其资料为：\n{text2}'
#     print(myprompt)
#     print('------------------------------')
#     print('大模型开始工作')
#     print('------------------------------')
#     print(model_selection)
#     print(list(model_selection))
#     if model_selection != 'ChatGLM3-6B':
#         partial_message = ""
#         for response in client.generate(model=model_selection, prompt=myprompt, options={'temperature': temperature, },
#                                         stream=True, ):
#             content = response['response']
#             print(content, end='', flush=True)

#             partial_message += content
#             # print("partial_message================",partial_message)
#             # print("[[None, partial_message]================",[[None, partial_message]])
#             yield [[None, partial_message]]

#     else:
#         class StopOnTokens(StoppingCriteria):
#             def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#                 stop_ids = [29, 0]
#                 for stop_id in stop_ids:
#                     if input_ids[0][-1] == stop_id:
#                         return True
#                 return False

#         history_transformer_format = [[myprompt, ""]]
#         stop = StopOnTokens()

#         messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
#                             for item in history_transformer_format])
#         tokenizer = AutoTokenizer.from_pretrained("/home/LLM/chatglm3-6b/", trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained("/home/LLM/chatglm3-6b/",
#                                                      torch_dtype=torch.float16, device_map="auto",
#                                                      trust_remote_code=True)
#         model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
#         streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
#         generate_kwargs = dict(
#             model_inputs,
#             streamer=streamer,
#             max_new_tokens=1024,
#             do_sample=True,
#             top_p=0.95,
#             top_k=1000,
#             temperature=temperature + 0.001,
#             num_beams=1,
#             stopping_criteria=StoppingCriteriaList([stop])
#         )
#         t = Thread(target=model.generate, kwargs=generate_kwargs)
#         t.start()

#         partial_message = ""
#         for new_token in streamer:
#             if new_token != '<':
#                 partial_message += new_token
#                 yield [[None, partial_message]]




def mode_change(use_pattern):
    if use_pattern == '跨网域匹配':
        plat1 = '电信网'
        plat2 = '微博'
    if use_pattern == '跨平台匹配':
        plat1 = '微博'
        plat2 = '今日头条'
    if use_pattern == '同平台匹配':
        plat1 = '微博'
        plat2 = '微博'
    return plat1, plat2


def plat_change1(use_pattern, plat1, plat2):
    if use_pattern == '同平台匹配':
        plat2 = plat1

    user_name = gr.Dropdown(label=f"待匹配用户名（{plat1}）平台",
                            # value='老杨ysh_73',
                            allow_custom_value=True,
                            choices=['老杨ysh_73', 'znl李先生', 'SOD_大宝', '冷温柔Triste998', '禅是禅非', 'Bigg宇宙',
                                     '陈曦guitar',
                                     '马丽lili', '天翼帐号11018428363', 'G友116987631', '炫铃用户2911236035',
                                     '多米音乐_8196567',
                                     '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213', '王泽Happy',
                                     '夕阳beautiful'])
    return plat2, user_name


def plat_change2(use_pattern, plat1, plat2):
    # if use_pattern=='同平台匹配':
    #     plat1=plat2
    if plat2=='今日头条':
        user_name = gr.Dropdown(
                                allow_custom_value=True,
                                choices=['鹤壁新闻网', '鹤壁经济广播', '鹤壁第一深情', '中国新闻网',
                                         '云淡风轻2008lf', '海阔天空889999', '浙江日报',
                                         '清风FUFACE', '云淡风轻0826_813',
                                         '阿杰729', '幸福5705394745',])
                                         # '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213',
                                         # '王泽Happy', '夕阳beautiful'])
    else:
        user_name = gr.Dropdown(#label=f"待匹配用户名（{plat1}）平台",
                                # value='老杨ysh_73',
                                allow_custom_value=True,
                                choices=['老杨ysh_73', 'znl李先生', 'SOD_大宝', '冷温柔Triste998', '禅是禅非',
                                         'Bigg宇宙',
                                         '陈曦guitar',
                                         '马丽lili', '天翼帐号11018428363', 'G友116987631', '炫铃用户2911236035',
                                         '多米音乐_8196567',
                                         '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213', '王泽Happy',
                                         '夕阳beautiful'])
    match_name = gr.Dropdown(label=f"匹配成功的用户名（{plat2}）平台", interactive=True, allow_custom_value=False)
    return user_name, match_name
