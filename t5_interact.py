#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：chat_t5.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/5/6 11:26 
'''

import transformers
import torch
import os
import json
import random
import numpy as np
import argparse


def set_args():
    """     Sets up the arguments.     """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--work_mode', default='Train', type=str, required=True, help='生成设备')
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.005, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def interact(  args ):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    import torch
    from torch import cuda
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(      "mxmax/Chinese_Chat_T5_Base")
    model = AutoModelForSeq2SeqLM.from_pretrained(  "mxmax/Chinese_Chat_T5_Base")
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)

    def postprocess(text):
        return text.replace(".", "").replace('</>', '')

    def answer_fn(text, top_k=50):
        encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=args.max_len, return_tensors="pt").to(device)
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=args.max_len,
                             temperature=args.temperature, do_sample=True, repetition_penalty=args.repetition_penalty,
                             top_k=args.topk)
        result = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
        return postprocess(result[0])

    while True:
        text = input('请输入问题:')
        result = answer_fn(text, top_k=50)
        print("【<*_*>】(模型生成):", result)
        print('*' * 100)

# def train(args):
#     from t5_train import T5Trainer
#     import  pandas as pd
#
#     df = pd.read_csv( args ,sep='||')
#
#     T5Trainer(
#         dataframe       =df,
#         source_text     ="input",
#         target_text     ="target",
#         model_params    =model_params,
#         output_dir      ="./outputs",
#     )
#
#     return


def main():
    args = set_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    interact(args)
    return

if __name__ == '__main__':
    main(   )
    exit()
    
  
  