#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：zero_train.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/3/4 14:01 
'''
from transformers import GPT2Model
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live;
model = GPT2Model.from_pretrained("IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese");
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)

if __name__ == '__main__':
    exit()
    
  
  