#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：test_belle.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/5/7 11:57 
'''
if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sys

    model_path = "./BELLE-7B-2M/"  # You can modify the path for storing the local model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Human:")
    line = input()
    while line:
        inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85, temperature=0.35, repetition_penalty=1.2)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("Assistant:\n" + rets[0].strip().replace(inputs, ""))
        print("\n------------------------------------------------\nHuman:")
        line = input()

    exit()
    
  
  