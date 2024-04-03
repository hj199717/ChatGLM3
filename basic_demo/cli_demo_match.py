import os
import platform
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pymysql
from sqlalchemy import create_engine  # sqlalchemy版本指定为1.4.39


def find_match_user_name(user_name='老杨ysh_73'):
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
        pass
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
                 'followers_count', 'friends_count','description',
                 'verified_reason']]
        df.fillna('无', inplace=True)
        gender_dic = {'m': '男', 'f': '女'}
        df['gender'] = df.apply(lambda x: gender_dic.get(x.gender, '未知'), axis=1)
        user_info = df.iloc[0, :].tolist()
        text = f'用户昵称为“{user_info[0]}”，该用户的uid为{user_info[1]}，该用户的性别为{user_info[2]}，该用户的地理位置为{user_info[3].replace(" ", "")}，\
该用户的等级为{user_info[4]}级，该用户的发帖总数为{user_info[5]}，该用户的粉丝数为{user_info[6]}，该用户的关注数为{user_info[7]}，\
该用户的用户个人描述为“{user_info[8]}”，该用户的认证原因为“{user_info[9]}”'
        return text

    text1=transfer_df(df1)
    text2=transfer_df(df2)
    return text1,text2

#----------------------------------------------------------

MODEL_PATH = os.environ.get('MODEL_PATH', r'/home/LLM/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# tokenizer = AutoTokenizer.from_pretrained(r"/home/LLM/chatglm3-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained(r"/home/LLM/chatglm3-6b", trust_remote_code=True, device='cuda')

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main(prompt_1st=''):
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    tag=False
    if prompt_1st:
        tag=True
    while True:
        if tag:
            query=prompt_1st
            tag=False
        else:
            query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":

    user1 = '老杨ysh_73'
    user2 = find_match_user_name(user1)
    text1, text2 = find_2_user_detail(user1, user2)
    myprompt = f'''现有两个微博用户账号的资料如下：
第一个用户的资料为：{text1}\n
第二个用户的资料为：{text2}\n
现在我们已经确认这两个微博用户为同一人的两个不同账号，请通过这两个用户的资料分析两个用户属于同一人的原因，并生成详细的分析报告，并生成详细的分析报告'''
    print("用户：",myprompt)

    main(myprompt)