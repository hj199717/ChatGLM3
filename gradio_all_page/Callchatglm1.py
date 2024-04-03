from transformers import AutoTokenizer, AutoModel
import gradio as gr
import os
from prompts_tests import extract_and_concatenate_values, extract_user_data, extract_quotes_content
from user_matching import *
import re

# from retrieval_test import langtrainembeding

# test=langtrainembeding()
path = "kgs3.json"


# tokenizer = AutoTokenizer.from_pretrained("/home/LLM/chatglm3-6b/", trust_remote_code=True)
# model = AutoModel.from_pretrained("/home/LLM/chatglm3-6b/", trust_remote_code=True).half().cuda()
#
# model = model.eval()


def add_text(history, text):
    history = history + [(text, None)]
    print("---------", history, gr.update(value="", interactive=False))
    return history, gr.update(value="", interactive=False)


def bot(history):
    """
    聊天调用的函数
    :param history:
    :return:
    """
    message = history[-1][0]
    print(history)

    if isinstance(message, tuple):
        response = "文件上传成功！！"
    else:
        text = history[-1][0]
        print("==============================", text)
        user = extract_quotes_content(text)[0]
        print(user)
        kg_one, kg_attr = extract_user_data(path, user)

        kg_one = re.sub(r'\\', '', str(kg_one))
        kg_location = extract_and_concatenate_values(str(kg_one), '地理位置')
        kg_content = extract_and_concatenate_values(str(kg_one), '发文内容')
        kg_emotion = extract_and_concatenate_values(str(kg_one), '发文倾向性')
        kg_aite = extract_and_concatenate_values(str(kg_one), '提及好友')
        #
        text_attr = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{}，回答问题：请输出文本{}中的对应字段，作为用户的基本属性信息""".format(
            kg_attr, kg_attr)
        text_id = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{}，回答问题：请计算文本{}中，'用户id'出现的次数，只需要回答一个数值大小即可，不需要其他内容""".format(
            kg_one, kg_one)
        text_location = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{}，回答问题：请计算文本{}中，'用户id'出现的次数，只需要回答一个数值大小即可，不需要其他内容""".format(
            kg_location, kg_location)
        text_content = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{},回答问题：请概括一下{}的关键词，输出最具代表性的3个关键词""".format(
            kg_content, kg_content)
        text_emotion = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{}，回答问题：请问{}中的内容是什么？，不需要其他内容""".format(
            kg_emotion, kg_emotion)
        text_aite = """请根据以下知识完成回答，回答内容必须严格遵守知识约束:{}，回答问题：请问{}中的内容是什么？""".format(
            kg_aite, kg_aite)

        #
        response_id = model.chat(tokenizer, text_id, history=[])[0]
        response_location = model.chat(tokenizer, text_location, history=[])[0]
        response_content = model.chat(tokenizer, text_content, history=[])[0]
        response_emotion = model.chat(tokenizer, text_emotion, history=[])[0]
        response_aite = model.chat(tokenizer, text_aite, history=[])[0]

        texts = """请主要参考提供以下知识完成回答，回答内容必须严格遵守知识约束：以下是用户"{}"的属性信息{}，近期的微博行为数据:"用户地理位置：{}，用户发文次数：{},发文关键词：{},发文倾向性为：{}，提及好友为：{}"，请根据上述知识，回答问题：{}，
        只需要输出用户性别，用户等级，粉丝数，关注数，互关数，创建时间，是否认证，认证类型，用户地理位置，用户发文次数，发文关键词，发文倾向，提及好友是谁即可，如果没有，直接回答无即可。并基于上述信息，结合政治方面，对用户进行深层次分析""".format(
            user, text_attr, response_location, response_id, response_content, response_emotion, response_aite, text)

        history[-1][1] = model.chat(tokenizer, texts, history=[])[0]

    yield history


# with gr.Blocks(css=".gradio-interface {height: 1400px;") as demo:
#
#     with gr.Row(variant="panel",elem_classes=".gradio-interface {height: 1400px;"):
#
#         introduction = gr.Textbox(lines=10, label="SongshanLab", default_value="nihao")
#
#         with gr.Column(scale=4):
#             # chatbot = gr.Chatbot(label='SongshanLab',height=1000,bubble_full_width=False)
#             chatbot = gr.Chatbot(
#
#                 [],
#                 title="SongshanLab",
#                 # height=1000,
#                 bubble_full_width=False,
#                 avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
#                 )
#             with gr.Row():
#                 txt = gr.Textbox(
#                     scale=10,
#                     show_label=False,
#                     placeholder="Enter text and press enter, or upload an image",
#                     container=False,
#                 )
#                 btn = gr.UploadButton("📁", file_types=['txt'])
#
#     txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
#         bot, chatbot, chatbot
#     )
#
#     txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
def process_csv(file):
    # 使用 Pandas 读取上传的 CSV 文件并转换为 DataFrame
    df = pd.read_csv(file.name)
    # 返回 DataFrame
    return df


with gr.Blocks(css=".gradio-interface {height: 1400px;") as demo:
    gr.Markdown("<center><strong> <font color=grey size=5> 嵩山实验室用户画像大模型系统 </font><strong> </center>")
    with gr.TabItem("用户画像系统"):
        with gr.Row():
            with gr.Column():
                # radio = gr.Radio(
                # ["用户画像功能", "用户画像定制功能"], label="请选择功能")
                img = gr.Image("songshanlogo.png", label='logo', height=200)
                text = gr.Textbox(lines=4, label="用户画像系统介绍", interactive=True,
                                  value="定制用户画像功能，可以根据用户的需求，结合知识库中的数据，对用户画像进行预测。")
                # radio.change(fn=change_textbox, inputs=radio, outputs=text)
                bar = gr.Slider(label='大模型温度', value=0.1, minimum=0, maximum=1, step=0.05, container=True,
                                interactive=True)
                models = gr.Dropdown(label='大模型选择', value='gemma:7b-instruct-fp16', allow_custom_value=True,
                                     choices=['gemma:7b-instruct-fp16',
                                              # 'gemma:7b-text-fp16',
                                              'gemma:7b']
                                     , interactive=True)
                changess = gr.CheckboxGroup(
                    ["水军倾向", "辱骂倾向", "信访倾向", "诈骗倾向", "追星倾向", "购物倾向", "政务倾向", "网暴倾向"],
                    label="用户画像定制", info="可以选择按钮，定制用户画像预测")
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(label='SongshanLab', height=700, bubble_full_width=False,
                                     avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))), )
                with gr.Row():
                    txt = gr.Textbox(
                        scale=10,
                        show_label=False,
                        placeholder="请输入用户昵称进行用户画像预测...",
                        container=False,
                    )
                    btn = gr.UploadButton("📁", file_types=['txt'])
                    user_csv = gr.DataFrame(visible=False)
                    btn.click(process_csv, [btn], )

        # text_button.click(flip_text, inputs=text_input, outputs=text_output)

        txt_msg = txt.submit(add_text, [chatbot, txt, changess], [chatbot, txt], queue=False).then(
            bot, chatbot, chatbot)

        txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    with gr.TabItem("用户匹配系统"):
        # print(gr.__version__)
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
                                    choices=['微博', '今日头条', '抖音', 'B站', '电信网'])
                plat2 = gr.Dropdown(label='平台2（匹配平台）', value='微博', allow_custom_value=False,
                                    choices=['微博', '今日头条', '抖音', 'B站', '电信网'])
                use_pattern.change(mode_change, use_pattern, [plat1, plat2])

                temperature = gr.Slider(label='大模型温度', value=0, minimum=0, maximum=1, step=0.05, container=True)
                top_p = gr.Slider(label='大模型top_p', value=0, minimum=0, maximum=1, step=0.05, container=True)
                model = gr.Dropdown(label='大模型选择', value='gemma:7b-instruct-fp16', allow_custom_value=True,
                                    choices=['gemma:7b-instruct-fp16',
                                             # 'gemma:7b-text-fp16',
                                             'qwen:72b',
                                             'ChatGLM3-6B',
                                             'gemma:7b'])
            with gr.Column(scale=8):
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
                match_button.click(find_match_user_list, [user_name, plat2], [match_name])
                llm_res = gr.Chatbot(label="用户匹配分析报告", height=650)
                llm_button.click(detailed_report, [model, temperature, top_p, user_name, match_name, plat1, plat2],
                                 [llm_res])
                user_name.change(find_user_pd, [user_name, plat1], user_detail)
                match_name.change(find_user_pd, [match_name, plat2], match_user_detail)
                # llm_res.postprocess('123')
                plat1.change(plat_change1, [use_pattern, plat1, plat2], [plat2, user_name])
                plat2.change(plat_change2, [use_pattern, plat1, plat2], [user_name, match_name])

                # chatbot = gr.Chatbot(label='SongshanLab', height=450, bubble_full_width=False)
                # message = gr.Textbox(label='请输入问题: ')
                # clear_history = gr.Button("清除")
                # send = gr.Button("发送")

demo.queue()
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860)
