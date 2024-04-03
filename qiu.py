# -*-coding:utf-8 -*-
'''
File       : qiu.py
Time       : 2024/3/14 8:49
Author     : He Jia
version    : 1.0.0
Description: 
'''
import tkinter as tk

# 创建主窗口
root = tk.Tk()

# 创建标签
label = tk.Label(root, text="请输入您的问题:")
label.pack()

# 创建文本框
text_box = tk.Text(root, height=10)
text_box.pack()

# 创建按钮
button = tk.Button(root, text="提交", command=lambda: answer_question())
button.pack()

# 创建答案标签
answer_label = tk.Label(root, text="")
answer_label.pack()

# 监听用户输入
def answer_question():
    # 获取用户输入
    question = text_box.get("1.0", tk.END)

    # 回答问题
    answer = "答案：..."

    # 显示答案
    answer_label["text"] = answer

# 启动主窗口
root.mainloop()
