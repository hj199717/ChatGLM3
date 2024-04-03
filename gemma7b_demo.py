# -*-coding:utf-8 -*-
'''
File       : gemma7b_demo.py
Time       : 2024/3/13 16:48
Author     : He Jia
version    : 1.0.0
Description: 
'''
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/home/LLM/gemma-7b")
model = AutoModelForCausalLM.from_pretrained("/home/LLM/gemma-7b", device_map="auto",max_length=512)

input_text = "养猫的好处有哪些。"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

print('----------------------------------------')

input_text = '''## 已知以下两个微博用户隶属同一自然人，两个微博用户的详细资料如下所示，请分析两个用户的相似性，并指出两个微博用户隶属同一自然人的线索和原因，生成分析报告。
## 第一个用户的资料为：用户昵称为“老杨ysh_73”，该用户的uid为1918358391，该用户的性别为男，该用户的地理位置为天津红桥区，该用户的等级为0级，该用户的发帖总数为4488，该用户的粉丝数为5849，该用户的关注数为75，该用户的用户个人描述为“三个方法可以解决所有的问题：接受，改变，离开。”，该用户的认证原因为“”
## 第二个用户的资料为：用户昵称为“老杨zc”，该用户的uid为2158902822，该用户的性别为男，该用户的地理位置为山东济宁，该用户的等级为0级，该用户的发帖总数为817，该用户的粉丝数为504，该用户的关注数为1863，该用户的用户个人描述为“”，该用户的认证原因为“”'''
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))