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
client = Client(host='172.16.30.181:30620')

#print(gr.__version__)
def illustrate_pattern(pattern):

    if pattern=="跨网域匹配":
        return "跨网域匹配"
    elif pattern=="跨平台匹配":
        return "跨平台匹配"
    elif pattern=="同平台匹配":
        return "同平台匹配"
    else:
        return ""

def find_2_user_detail(user1, user2):
    table_name = 'suzhou1st'  # 处理原表名称
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'weibo'
    mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
    con = create_engine(f'{mysql_ip}{database}')
    df1 = pd.read_sql(f"select * from {table_name} where screen_name='{user1}'", con)
    print(df1)
    print('---------------------------------')
    df2 = pd.read_sql(f"select * from {table_name} where screen_name='{user2}'", con)
    print(df2)
    print('-------------------------------------------')
    print(df2.columns)

    def transfer_df(df: pd.DataFrame):
        df = df[['screen_name', 'uid', 'gender', 'location', 'mbrank', 'post_count',
                 'followers_count', 'friends_count', 'description',
                 'verified_reason']]
        df.fillna('无', inplace=True)
        gender_dic = {'m': '男', 'f': '女'}
        df['gender'] = df.apply(lambda x: gender_dic.get(x.gender, '未知'), axis=1)
        user_info = df.iloc[0, :].tolist()
        text = f'用户昵称为“{user_info[0]}”，该用户的uid为{user_info[1]}，该用户的性别为{user_info[2]}，该用户的地理位置为{user_info[3].replace(" ", "")}，\
该用户的等级为{user_info[4]}级，该用户的发帖总数为{user_info[5]}，该用户的粉丝数为{user_info[6]}，该用户的关注数为{user_info[7]}，\
该用户的用户个人描述为“{user_info[8]}”，该用户的认证原因为“{user_info[9]}”。'
        return text

    text1 = transfer_df(df1)
    text2 = transfer_df(df2)
    return text1, text2
def find_match_user_name(user_name):
    # 此处调整MySQL相关参数
    table_name = 'net_match'  # 处理原表名称
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'net_match'
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

        match_user = match_lst[0]
        return match_user

def detailed_report(user1, user2):
    text1, text2 = find_2_user_detail(user1, user2)
    myprompt = f'''## 现在已经确认以下两个微博用户隶属同一自然人，两个微博用户的详细资料如下所示，请分析两个用户的相似性，并指出两个微博用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。
            第一个用户的资料为：{text1}
            第二个用户的资料为：{text2}'''
    print(myprompt)
    print('------------------------------')
    print('大模型开始工作')
    print('------------------------------')

    partial_message = ""
    for response in client.generate(model='gemma:7b', prompt=myprompt, options={'temperature': 0, }, stream=True, ):
        content = response['response']
        print(content, end='', flush=True)

        partial_message += content
        yield [[None,partial_message]]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            img = gr.Image("songshanlogo.png",label='logo',height=200)
            use_pattern = gr.Radio([
                    '跨网域匹配',
                    '跨平台匹配',
                    '同平台匹配'],
                label="匹配模型分析模式",
                value='同平台匹配',
                interactive=True)
            inter = gr.Interface(
                illustrate_pattern,
                inputs=[use_pattern],
                outputs=[gr.Textbox(label="匹配模型使用说明:",lines=5,value="同平台匹配")],
                live=True,
                allow_flagging="never"
            )

        with gr.Column(scale=7):
            with gr.Row():
                user_name=gr.Textbox(label="待匹配用户名",value='老杨ysh_73')

                match_button = gr.Button("进行匹配")
                match_name = gr.Textbox(label="匹配成功的用户名")
                match_button.click(find_match_user_name,user_name,match_name)
                llm_button = gr.Button("查看详细分析报告")


            llm_res = gr.Chatbot(label="用户匹配分析报告",height=550)
            llm_button.click(detailed_report, [user_name,match_name], [llm_res])
            #llm_res.postprocess('123')

            # chatbot = gr.Chatbot(label='SongshanLab',height=450,bubble_full_width=False)
            # message = gr.Textbox(label='请输入问题: ')
            # clear_history = gr.Button("清除")
            # send = gr.Button("发送")
demo.launch(server_name='0.0.0.0',server_port=7860)
