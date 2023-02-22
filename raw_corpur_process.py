#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：dialogGPT2_RHFL 
@File    ：raw_corpur_process.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2023/2/23 0:59 
'''

def process(save_path,df):
    f = open(save_path, 'w')
    cnt = 0
    for index, row in df.iterrows():
        key   = row['title']
        value = row['reply']
        f.write( key.strip().replace(' ', '') + '\n'    )
        f.write( str(value).strip().replace(' ', '') + '\n'  )
        f.write('\n\n')
    f.close()

    return


import  pandas as pd
if __name__ == '__main__':

    #df = pd.read_csv('F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv')
    df = pd.read_csv('F:\\聊天机器人\\Q-A-matching\\data\\baoxian_zhidao\\baoxianzhidao_filter.csv')
    print(  len(df)    )
    print(  df.columns )
    print( df.head(5)  )
    process('F:\\聊天机器人\\baoxian_gpt2.csv',df)
    exit()
    
  
  