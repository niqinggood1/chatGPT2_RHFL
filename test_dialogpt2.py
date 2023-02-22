#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：test_dialogpt2.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/2/22 13:56 
'''
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
# from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    # parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
    #                     help='模型参数')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()

def create_logger(args):
    """    将日志输出到日志文件和控制台     """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger



from transformers import AutoTokenizer
def main():
    args = set_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cpu'#'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储

    print('# 1 #' * 5)
    history = []
    text = "你是谁"
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    text_ids =  tokenizer.encode(text, add_special_tokens=False)
    history.append(text_ids)
    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

    for history_id, history_utr in enumerate(history[-args.max_history_len:]):
        input_ids.extend(history_utr)
        input_ids.append(tokenizer.sep_token_id)
    input_ids = torch.tensor(input_ids).long().to(device)
    input_ids = input_ids.unsqueeze(0)

    response = model.generate(input_ids, max_new_tokens=16)
    text = tokenizer.convert_ids_to_tokens(response.squeeze())
    print('outputs1:', text)
    print('outputs1:', text[-16:])



    print( '# 2 #'*5 )
    text = "你是谁"
    gpt2_tokenizer = AutoTokenizer.from_pretrained( "uer/gpt2-chinese-cluecorpussmall" )
    gpt2_tokenizer.eos_token = gpt2_tokenizer.pad_token
    input_ids = gpt2_tokenizer.encode(text) ;
    print('input_ids:',input_ids) ;
    print('input_:',   gpt2_tokenizer.convert_ids_to_tokens(  input_ids  )  )
    input_ids = torch.tensor( input_ids ).long().to(device)
    response = model.generate( input_ids.unsqueeze(dim=0),max_new_tokens=16  )
    print('response:',response )
    text = gpt2_tokenizer.convert_ids_to_tokens(  response.squeeze()  )
    print('outputs:', text      )
    print('outputs:',text[-16:] )


if __name__ == '__main__':
    main()

  
  