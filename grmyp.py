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


def find_user_pd(user1):
    table_name = 'suzhou1st'  # 处理原表名称
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'weibo'
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
    df.columns = ['昵称', 'uid编号', '性别', '位置', '等级', '发帖数', '粉丝数', '关注数', '个人描述', '认证原因']
    return df


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


def find_match_user_list(user_name):
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
        table_colomn_2 = gr.Dropdown(value=match_lst[0], choices=match_lst)
        return table_colomn_2
        # match_name.choices=match_lst
        # match_user = match_lst[0]
        # return match_user
        # return match_lst[0]


def detailed_report(model_selection, temperature, user1, user2):
    # yield [[None,None]]
    text1, text2 = find_2_user_detail(user1, user2)
    myprompt = f'''## 现在已经确认以下两个微博用户隶属同一自然人，两个微博用户的详细资料如下所示，请分析两个用户的相似性，并指出两个微博用户隶属同一自然人的线索和原因，生成分析报告，有具体数字和数据需要列出，报告越详细越好。
            第一个用户的资料为：{text1}
            第二个用户的资料为：{text2}'''
    print(myprompt)
    print('------------------------------')
    print('大模型开始工作')
    print('------------------------------')

    partial_message = ""
    for response in client.generate(model=model_selection, prompt=myprompt, options={'temperature': temperature, },
                                    stream=True, ):
        content = response['response']
        print(content, end='', flush=True)

        partial_message += content
        yield [[None, partial_message]]


def mode_change(use_pattern):
    if use_pattern == '跨网域匹配':
        plat1 = '电信网'
        plat2 = '微博'
    if use_pattern == '跨平台匹配':
        plat1 = '微博'
        plat2 = '头条'
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
    match_name = gr.Dropdown(label=f"匹配成功的用户名（{plat2}）平台", interactive=True, allow_custom_value=False)
    return match_name


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            img = gr.Image("songshanlogo.png", label='logo', height=200)
            use_pattern = gr.Radio([
                '跨网域匹配',
                '跨平台匹配',
                '同平台匹配'],
                label="匹配模型分析模式",
                value='同平台匹配',
                interactive=True,
                container=True,
                # scale=1
            )
            # inter = gr.Interface(
            #     illustrate_pattern,
            #     inputs=[use_pattern],
            #     outputs=[gr.Textbox(label="匹配模型使用说明:",lines=2,value="同平台匹配")],
            #     live=True,
            #     allow_flagging="never"
            # )
            plat1 = gr.Dropdown(label='平台1（原始平台）', value='微博', allow_custom_value=False,
                                choices=['微博', '头条', '抖音', 'B站', '电信网'])
            plat2 = gr.Dropdown(label='平台2（匹配平台）', value='微博', allow_custom_value=False,
                                choices=['微博', '头条', '抖音', 'B站', '电信网'])
            use_pattern.change(mode_change, use_pattern, [plat1, plat2])

            bar = gr.Slider(label='大模型温度', value=0.1, minimum=0, maximum=1, step=0.05, container=True)
            model = gr.Dropdown(label='大模型选择', value='gemma:7b-instruct-fp16', allow_custom_value=True,
                                choices=['gemma:7b-instruct-fp16',
                                         # 'gemma:7b-text-fp16',
                                         'gemma:7b'])

        with gr.Column(scale=9):
            with gr.Group():
                with gr.Row():
                    # user_name=gr.Textbox(label="待匹配用户名",value='老杨ysh_73')
                    user_name = gr.Dropdown(label="待匹配用户名（微博）平台",
                                            # value='老杨ysh_73',
                                            allow_custom_value=True,
                                            choices=['老杨ysh_73', 'znl李先生', 'SOD_大宝', '冷温柔Triste998',
                                                     '禅是禅非', 'Bigg宇宙', '陈曦guitar',
                                                     '马丽lili', '天翼帐号11018428363', 'G友116987631',
                                                     '炫铃用户2911236035', '多米音乐_8196567',
                                                     '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213',
                                                     '王泽Happy', '夕阳beautiful'])

                    match_button = gr.Button("进行匹配")
                    # match_name = gr.Textbox(label="匹配成功的用户名")
                    match_name = gr.Dropdown(label="匹配成功的用户名（微博）平台", interactive=True,
                                             allow_custom_value=False)

                    llm_button = gr.Button("查看详细分析报告")

                with gr.Row():
                    with gr.Accordion('查看用户详细资料', open=False):
                        user_detail = gr.DataFrame()  # label='用户资料')
                    with gr.Accordion('查看匹配用户详细资料', open=False):
                        match_user_detail = gr.DataFrame()  # label='匹配用户资料')
            match_button.click(find_match_user_list, user_name, [match_name])
            llm_res = gr.Chatbot(label="用户匹配分析报告", height=650)
            llm_button.click(detailed_report, [model, bar, user_name, match_name], [llm_res])
            user_name.change(find_user_pd, user_name, user_detail)
            match_name.change(find_user_pd, match_name, match_user_detail)
            # llm_res.postprocess('123')
            plat1.change(plat_change1, [use_pattern, plat1, plat2], [plat2, user_name])
            plat2.change(plat_change2, [use_pattern, plat1, plat2], match_name)

demo.launch(server_name='0.0.0.0', server_port=7860)
# demo.launch()
