#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：belle_interact.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/5/6 16:10 
'''

import argparse
import torch
def set_args():
    """     Sets up the arguments.     """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--work_mode', default='Train', type=str, required=True, help='生成设备')
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=0.35, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0.85, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='Chinese_Chat_T5_Base', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_in_len', default=750, type=int, required=False, help='训练时，输入数据的最大长度')
    parser.add_argument('--max_len', type=int, default=300, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=1, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()

def main():
    args = set_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sys

    model_path = args.model_path # "./"  # You can modify the path for storing the local model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(  model_path)
    print("Human:")
    line = input()
    history=[  ]
    while line:
        inputs = 'Human: ' + line.strip() + '\n\nAssistant:'
        inputs2 = '' + inputs
        for i in range( 1,args.max_history_len+1 ):
            if i <len(history):
                if len( history[ -i ] ) + len( inputs2 )<= args.max_in_len:
                    inputs2 = history[ -i ] + inputs2
        history.append( inputs )

        # with_history_input =    '\n'.join( history[-args.max_history_len:] ) + inputs
        # if len( with_history_input )<args.max_len:
        #     inputs = inputs

        input_ids = tokenizer(inputs2, return_tensors="pt").input_ids
        outputs   = model.generate(input_ids, max_new_tokens=args.max_len, do_sample=True, top_k=args.topk, top_p=args.topp,
                                 temperature=args.temperature, repetition_penalty=args.repetition_penalty )
        #outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85, temperature=0.35, repetition_penalty=1.2)
        rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("Assistant:\n" + rets[0].strip().replace(inputs2, ""))

        print("\n------------------------------------------------\nHuman:")
        line = input()

if __name__ == '__main__':
    main()
    exit()
    
  
  