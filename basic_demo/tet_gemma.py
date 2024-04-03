from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(r'/home/LLM/gemma-7b-it')
model = AutoModelForCausalLM.from_pretrained(r'/home/LLM/gemma-7b-it', device_map="auto")

input_text = ""
while input_text not in ["quit", "exit", "clear"]:
    print('用户：',end='')
    input_text = input()
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    print('gemma-7b-it：')
    print(tokenizer.decode(outputs[0]))
    print('--------------------------------')