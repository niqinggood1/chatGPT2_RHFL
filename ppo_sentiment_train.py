# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

PPO + GPT2, 中文情感分析。

Author: pankeyu
Date: 2022/12/27
"""
import time
import random

import torch
from rich import print
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from trl.gpt2 import GPT2HeadWithValueModel


from iTrainingLogger import iSummaryWriter

config = {
    "steps": 20000,
    "batch_size":6,
    "forward_batch_size":2, #16
    "ppo_epochs": 8,
    "lr": 1.41e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 16
}

from transformers import GPT2LMHeadModel
def get_ppo_basemodel( config, arg  ):
    writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment-Zh')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe_device = 0 if torch.cuda.is_available() else -1
    print( 'arg model path',arg.model_path )
    gpt2_model                  =  GPT2HeadWithValueModel.from_pretrained( arg.model_path ) #GPT2HeadWithValueModel.from_pretrained( model_path=arg.model_path )
    gpt2_model_ref              =  GPT2HeadWithValueModel.from_pretrained( arg.model_path ) #GPT2HeadWithValueModel.from_pretrained( model_path=arg.model_path )

    gpt2_model.transformer      = GPT2LMHeadModel.from_pretrained( arg.model_path)
    gpt2_model_ref.transformer  = GPT2LMHeadModel.from_pretrained( arg.model_path)

    gpt2_tokenizer              = AutoTokenizer.from_pretrained( arg.vocab_path,sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]" )  #(config['model_name'])
    gpt2_tokenizer.eos_token    = gpt2_tokenizer.pad_token
    gpt2_model.to(device)
    gpt2_model_ref.to(device)

    return   gpt2_model,gpt2_model_ref, iSummaryWriter  #ppo_trainer,



def train_ppo_network( ppo_trainer, gpt2_tokenizer, batch, query_tensors ,response_tensors,rewards,device,writer,epoch,gen_len=500  ):
    logs, timing = dict(), dict()
    t0 = time.time()
    t   = time.time()

    timing['time/get_response'] = time.time() - t
    t = time.time()
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    rewards = torch.tensor(rewards,dtype=torch.float32).to(device)
    print('in train_ppo_network rewards:',rewards )

    ppo_trainer.model.train()
    ppo_trainer.ref_model.train()

    stats = ppo_trainer.step(query_tensors, response_tensors,rewards)  # PPO Update

    gen_kwargs = {
        "min_length": -1,
        "top_k": 1,
        # "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": gpt2_tokenizer.eos_token_id
    }
    ppo_trainer.model.eval()
    findk=0
    for i in range( len(rewards) ):
        if rewards[i] <0:
            response        = ppo_trainer.model.generate( query_tensors[i].unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs)
            final_responese = response.squeeze()[-gen_len:]
            text = gpt2_tokenizer.decode(final_responese.squeeze())
            print('check:', '[query]:%s'%gpt2_tokenizer.decode( query_tensors[i]),'[response]>>%s'%gpt2_tokenizer.decode( response_tensors[i]) )
            print( 'After train ','[query]:%s'%batch['query'][i],'[response]>>%s'%text  )


    timing['time/optimization'] = time.time() - t
    timing['time/epoch'] = time.time() - t0  # logging
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")

    # print('Random Sample 5 text(s) of model output:')
    # for i in range(5):  # 随机打5个生成的结果
    #     print(f'{i + 1}. {random.choice(texts)}')
    # writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    # writer.add_scalar('ppo/loss/policy', stats['ppo/loss/policy'], epoch)
    # writer.add_scalar('ppo/loss/value', stats['ppo/loss/value'], epoch)
    # writer.add_scalar('ppo/policy/entropy', stats['ppo/policy/entropy'], epoch)
    # writer.add_scalar('ppo/policy/policykl', stats['ppo/policy/policykl'], epoch)
    # writer.record()
    return ppo_trainer

if __name__ == '__main__':
    ppo_sentiment_train( config,prompts)