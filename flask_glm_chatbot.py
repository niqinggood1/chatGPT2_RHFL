#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：GPT2-chitchat
@File    ：flask_gpt_chatbot.py
@IDE     ：PyCharm
@Author  ：patrick
@Date    ：2023/2/1 16:13
'''
from flask import Flask, request, Response
from flask import json

app = Flask(__name__)
import argparse


def set_args():
    """     Sets up the arguments.      """
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--repetition_penalty', default=1.06, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--ppo_train', action='store_true', help='默认为False不进行PPO训练')
    parser.add_argument('--save_model_path', default='model', type=str, required=False,
                        help='模型输出路径')

    parser.add_argument('--host', default='0.0.0.0', type=str, required=False, help='wangwang windows picture')
    parser.add_argument('--port', default=6006, type=int, required=False, help='wangwang windows picture')

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


import logging
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer


def load_model(args):
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import sys

    model_path = args.model_path  # "./"  # You can modify the path for storing the local model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()
    return model, tokenizer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


import torch.nn.functional as F
import os

global model, tokenizer, args, global_map, global_key_value, batch
args = set_args()
model, tokenizer = load_model(args)
device = 'cuda' if args.cuda else 'cpu'
global_map = {}
global_key_value = {}

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_path = args.model_path # "./"  # You can modify the path for storing the local model
model       = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer   = AutoTokenizer.from_pretrained(  model_path)



def init_batch():
    batch = {}
    batch.setdefault('tokens', [])
    batch.setdefault('query', [])
    batch.setdefault('response', [])
    batch.setdefault('response_tensors', [])
    batch.setdefault('rewards', [])
    batch.setdefault('resp_idx', {})
    return batch


batch = init_batch()


@app.route('/reload_model___', methods=["get"])
def reload_model():
    global model, tokenizer, args
    model, tokenizer = load_model(args)
    print('model reload success ############################')
    return 'model reload succ'


from ppo_sentiment_train import get_ppo_basemodel, config, train_ppo_network
from trl.ppo import PPOTrainer



from os.path import join, exists

global epoch, feedbackcnt
epoch = 0;
feedbackcnt = 0;


@app.route('/feedback', methods=["get", "POST"])
def feedback():
    global ppo_trainer
    global global_map, global_key_value, args, epoch, feedbackcnt
    global batch

    feedbackcnt += 1
    if request.method == "GET":
        username = request.form['username']
        text = request.form['text']
    if request.method == "POST":
        re_data = request.get_data()
        request_json_data = json.loads(re_data)
        username = request_json_data['username']
        text = request_json_data['text']
        feedback = request_json_data['feedback']

    text = urllib.parse.unquote(text)
    print('feedback:', feedbackcnt, feedback, text)

    if args.ppo_train == False:
        print('不训练 ppo')
        return ''

    if text not in batch['resp_idx']:
        print('not found :', text)
        return ''
    return ''


import urllib


@app.route('/gpt_chat', methods=["POST"])
def gen_response():
    global global_map, global_key_value, args
    if request.method == "GET":
        username = request.form['username']
        query = request.form['text']
    if request.method == "POST":
        re_data = request.get_data()
        request_json_data = json.loads(re_data)
        username = request_json_data['username']
        query = request_json_data['text']

    history = global_map.setdefault(username, [])
    query = urllib.parse.unquote(query)
    text = query
    print('from user', username, 'text:', text, 'history sentens len:', len(history))

    inputs  = 'Human: ' + query.strip() + '\n\nAssistant:'

    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=args.max_len, do_sample=True, top_k=args.topk, top_p=args.topp,
                             temperature=args.temperature, repetition_penalty=args.repetition_penalty)
    # outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_k=30, top_p=0.85, temperature=0.35, repetition_penalty=1.2)
    rets = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print( "Assistant:\n" + rets[0].strip().replace(inputs, "") )

    # batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors ]
    return json.dumps({'message': 'Ok', 'success': 1, 'content': rets[0]})


if __name__ == '__main__':
    app.run(host=args.host, port=args.port, debug=True)
    exit()



# text_ids = tokenizer.encode(text, add_special_tokens=False)
    # history.append(text_ids)
    # input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
    #
    # for history_id, history_utr in enumerate(history[-args.max_history_len:]):
    #     input_ids.extend(history_utr)
    #     input_ids.append(tokenizer.sep_token_id)
    # # input_ids.extend(history_utr) ; input_ids.append(tokenizer.sep_token_id)
    #
    # input_ids = input_ids[-500:];
    # print('history sentence: ', tokenizer.decode(input_ids))
    # input_ids = torch.tensor(input_ids).long().to(device)
    #
    # input_ids2 = input_ids.unsqueeze(0)
    # response = []  # 根据context，生成的response
    # for _ in range(args.max_len):
    #     outputs = model(input_ids=input_ids2)
    #     logits = outputs.logits
    #     next_token_logits = logits[0, -1, :]
    #     # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
    #     for id in set(response):
    #         next_token_logits[id] /= args.repetition_penalty
    #     next_token_logits = next_token_logits / args.temperature
    #     # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    #     next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
    #     filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
    #     # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    #     next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    #     if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束 or  next_token ==tokenizer.eos_token
    #         break
    #     response.append(next_token.item())
    #     input_ids2 = torch.cat((input_ids2, next_token.unsqueeze(0)), dim=1)
    # history.append(response)
    #
    # global_map[username] = history
    # resp_text = tokenizer.convert_ids_to_tokens(response)
    # resp_text = "".join(resp_text)
    # print("chatbot:" + resp_text)
    # global_key_value.setdefault(resp_text, text)
    #
    # batch.setdefault('tokens', []).append(input_ids)  # ['tokens']
    # batch.setdefault('query', []).append(query)  # ['query']
    # batch.setdefault('response', []).append(resp_text)
    # batch.setdefault('response_tensors', []).append(response)
    # batch.setdefault('rewards', []).append(0.07)
    # batch.setdefault('resp_idx', {})
