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
# def illustrate_pattern(pattern):
#     if pattern == "跨网域匹配":
#         return "跨网域匹配"
#     elif pattern == "跨平台匹配":
#         return "跨平台匹配"
#     elif pattern == "同平台匹配":
#         return "同平台匹配"
#     else:
#         return ""


def find_user_pd(user1):
    table_name = 'ods_internet_user_data_sz'  # 处理原表名称
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'db_project'
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


def find_user_detail(user1):
    table_name = 'ods_internet_user_data_sz'  # 处理原表名称
    host = '172.16.30.216'
    port = 3306
    user = 'root'
    password = 'root'
    database = 'db_project'
    mysql_ip = f'mysql+pymysql://{user}:{password}@{host}:{port}/'
    con = create_engine(f'{mysql_ip}{database}')
    df = pd.read_sql(f"select * from {table_name} where screen_name='{user1}'", con)
    print(df)

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

    text = transfer_df(df)
    return text



def detailed_report(model_selection, temperature, user):
    # yield [[None,None]]
    text= find_user_detail(user)
    myprompt = f'''##已知一个微博用户的资料为：{text}\n##根据提供的微博用户数据。我想通过这些数据生成用户属性画像，希望了解用户的性别、年龄、兴趣爱好、职业等信息，同时生成尽可能准确的用户属性画像，可以考虑分析用户的微博内容、转发行为、关注信息等来推断用户的属性。生成的画像可以包括用户的偏好、兴趣爱好、所在地区、职业等。希望模型能够根据分析结果为每个用户生成尽可能准确的用户属性画像，以帮助我们更好地了解用户群体并进行个性化推荐等应用。谢谢！'''
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


# def mode_change(use_pattern):
#     if use_pattern == '跨网域匹配':
#         plat1 = '电信网'
#         plat2 = '微博'
#     if use_pattern == '跨平台匹配':
#         plat1 = '微博'
#         plat2 = '头条'
#     if use_pattern == '同平台匹配':
#         plat1 = '微博'
#         plat2 = '微博'
#     return plat1, plat2


# def plat_change1(use_pattern, plat1, plat2):
#     if use_pattern == '同平台匹配':
#         plat2 = plat1
#
#     user_name = gr.Dropdown(label=f"待匹配用户名（{plat1}）平台",
#                             # value='老杨ysh_73',
#                             allow_custom_value=True,
#                             choices=['老杨ysh_73', 'znl李先生', 'SOD_大宝', '冷温柔Triste998', '禅是禅非', 'Bigg宇宙',
#                                      '陈曦guitar',
#                                      '马丽lili', '天翼帐号11018428363', 'G友116987631', '炫铃用户2911236035',
#                                      '多米音乐_8196567',
#                                      '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213', '王泽Happy',
#                                      '夕阳beautiful'])
#     return plat2, user_name


# def plat_change2(use_pattern, plat1, plat2):
#     # if use_pattern=='同平台匹配':
#     #     plat1=plat2
#     match_name = gr.Dropdown(label=f"匹配成功的用户名（{plat2}）平台", interactive=True, allow_custom_value=False)
#     return match_name
def upload_file(file):
    # file_paths = [file.name for file in files]
    # return file_paths
    print(file)
    #with open(file, 'r',encoding='utf-8') as f:
    df=pd.read_csv(file,encoding='utf-8')
    print(df)
    return file

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2, min_width=200):
            img = gr.Image("songshanlogo.png", label='logo', height=200)
            use_pattern = gr.Radio([
                '用户画像',
                '用户属性画像',
                '用户行为画像'],
                label="用户画像分析模式",
                value='用户画像',
                interactive=True,
                container=True,
                # scale=1
            )

            plat = gr.Dropdown(label='平台', value='微博', allow_custom_value=False,
                                choices=['微博', '头条', '抖音', 'B站', '电信网'])

            #use_pattern.change(mode_change, use_pattern, [plat1, plat2])

            bar = gr.Slider(label='大模型温度', value=0.1, minimum=0, maximum=1, step=0.05, container=True)
            model = gr.Dropdown(label='大模型选择', value='gemma:7b-instruct-fp16', allow_custom_value=True,
                                choices=['gemma:7b-instruct-fp16',
                                         # 'gemma:7b-text-fp16',
                                         'gemma:7b'])
            upload_button = gr.UploadButton("'点击上传文件'",
                                            file_types=["text", ],
                                            # file_count="multiple"
                                            )
            file_output = gr.File(
                #height=160,
                file_types=["text",])

            upload_button.upload(upload_file, upload_button, file_output)

        with gr.Column(scale=9):
            with gr.Group():
                with gr.Row():
                    # user_name=gr.Textbox(label="待匹配用户名",value='老杨ysh_73')
                    user_name = gr.Dropdown(label="输入查询用户名",
                                            # value='老杨ysh_73',
                                            allow_custom_value=True,
                                            choices=['老杨ysh_73', 'znl李先生', 'SOD_大宝', '冷温柔Triste998',
                                                     '禅是禅非', 'Bigg宇宙', '陈曦guitar',
                                                     '马丽lili', '天翼帐号11018428363', 'G友116987631',
                                                     '炫铃用户2911236035', '多米音乐_8196567',
                                                     '鹤壁vivoXpiay', '贝壳手机用户2925697', '傲雪寒梅1213',
                                                     '王泽Happy', '夕阳beautiful'])

                    anal_button = gr.Button("生成画像")


                with gr.Row():
                    with gr.Accordion('查看用户详细资料', open=False):
                        user_detail = gr.DataFrame()  # label='用户资料')
                #     with gr.Accordion('查看匹配用户详细资料', open=False):
                #         match_user_detail = gr.DataFrame()  # label='匹配用户资料')

            llm_res = gr.Chatbot(label="用户匹配分析报告", height=650)
            anal_button.click(detailed_report, [model, bar, user_name], [llm_res])


            user_name.change(find_user_pd, user_name, user_detail)

            # llm_res.postprocess('123')
            # plat1.change(plat_change1, [use_pattern, plat1, plat2], [plat2, user_name])
            # plat2.change(plat_change2, [use_pattern, plat1, plat2], match_name)

demo.launch(server_name='0.0.0.0', server_port=7860)
# demo.launch()
