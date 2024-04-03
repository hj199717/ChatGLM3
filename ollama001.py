# -*-coding:utf-8 -*-
'''
File       : ollama001.py
Time       : 2024/3/14 9:52
Author     : He Jia
version    : 1.0.0
Description: 
'''
import ollama
from ollama import generate

while True:
  print('请输入问题')
  text=input()
  if text in ['cls','clear','stop','停止']:
    break
  # response = generate('gemma:7b', text)
  # print(response['response'])
  response = ollama.chat(model='gemma:7b', messages=[
    {
      'role': 'user',
      'content': text,
    },
  ])
  print('大模型的回答如下：')
  # print(response)
  print(response['message']['content'])
  #print(response['response'])
  print('-------------------------')