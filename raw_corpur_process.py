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
        f.write( str(key).strip().replace(' ', '') + '\n'    )
        f.write( str(value).strip().replace(' ', '') + '\n'  )
        f.write('\n\n')
    f.close()

    return


import  pandas as pd
if __name__ == '__main__':

    df = pd.DataFrame()
    import os,json

    path = 'F:\\聊天机器人\\问答知识库\\laws\\train'
    for file in os.listdir(path):
        tmp = json.loads( open(path + '\\' + file).read() ) ;
        df = pd.concat([df, pd.json_normalize( tmp ) ], axis=0, ignore_index=True)

    path = 'F:\\聊天机器人\\问答知识库\\laws\\test'
    for file in os.listdir(path):
        df = pd.concat([df, pd.json_normalize(json.loads( open(path + '\\' + file).read() ) )  ], axis=0, ignore_index=True)

    tmp_df = df[['question', 'answer']].rename(columns={'question': 'title', 'answer': 'reply'})
    process('F:\\聊天机器人\\%s.csv' % 'law', tmp_df)
    exit()

    #df = pd.read_csv('F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv')
    #medical
    import  json
    df  = pd.DataFrame()
    for line in open('F:\\聊天机器人\\问答知识库\\medical\medical.json'):
        # print('line:',line)
        js = json.loads(line)
        tmp_df = pd.json_normalize(js )
        df     = df.append(tmp_df)
    print('df',df.head(10))
    df.to_excel('./newdf1.xlsx')

    df_list =[]
    #是什么
    tmp_df  = df[['name','desc']].rename(columns={'name':'title', 'desc':'reply' } ); tmp_df['title'] = tmp_df['title']+'是什么'
    df_list.append( tmp_df )
    #如何阻止
    tmp_df = df[['name', 'prevent']].rename(columns={'name': 'title', 'prevent': 'reply'});tmp_df['title'] = tmp_df['title'] + '怎么预防'
    df_list.append(tmp_df)
    #花费 cost_money
    tmp_df = df[['name', 'cost_money']].rename(columns={'name': 'title', 'cost_money': 'reply'}); tmp_df['title'] = tmp_df['title'] + '治疗费用'
    df_list.append(tmp_df)
    #recommand_eat
    tmp_df = df[['name', 'recommand_eat']].rename(columns={'name': 'title', 'recommand_eat': 'reply'}); tmp_df['title'] = tmp_df['title'] + '推荐饮食'
    df_list.append(tmp_df)
    #cure_lasttime
    tmp_df = df[['name', 'cure_lasttime']].rename(columns={'name': 'title', 'cure_lasttime': 'reply'});tmp_df['title'] = tmp_df['title'] + '治疗时长'
    df_list.append(tmp_df)
    #symbol
    df['symptom'] = df['symptom'].apply(lambda x:  '、'.join(x) )
    tmp_df = df[['name', 'symptom']].rename(columns={'name': 'title', 'symptom': 'reply'}); tmp_df['title'] = tmp_df['title'] + '有哪些症状'
    df_list.append(tmp_df)

    tmp_df = df[['name', 'symptom']].rename(columns={ 'symptom': 'title','name': 'reply'}); tmp_df['title'] = '身体有'+tmp_df['title'] +'症状，可能是得了哪个冰'
    df_list.append(tmp_df)
    #cause
    tmp_df = df[['name', 'cause']].rename(columns={'name': 'title', 'cause': 'reply'});tmp_df['title'] = tmp_df['title'] + '得病原因可能是'
    df_list.append(tmp_df)

    #recommand_drug
    tmp_df = df[['name', 'recommand_drug']].rename(columns={'name': 'title', 'recommand_drug': 'reply'}); tmp_df['title'] = tmp_df['title'] + '推荐用药'
    df_list.append(tmp_df)
    #

    newdf = pd.concat(df_list,axis=0)

    # lines = open('F:\\聊天机器人\\问答知识库\\medical\medical.json').readlines()
    # df    = pd.json_normalize( lines )
    print(newdf.head(10))
    newdf.to_excel('./newdf.xlsx')


    df1 = pd.read_csv('F:\\聊天机器人\\问答知识库\\medical\\test_list.txt',sep='\t',header=None,names=['title','reply'])
    df2 = pd.read_csv('F:\\聊天机器人\\问答知识库\\medical\\train_list.txt',sep='\t',header=None,names=['title','reply'])
    df  = df1.append( df )
    df  = df1.append(df )
    process('F:\\聊天机器人\\%s.csv' % 'medical', df)
    exit()



    for csv in ['F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv',
                'F:\\聊天机器人\\Q-A-matching\\data\\baoxian_zhidao\\baoxianzhidao_filter.csv'
                ]:
        df = pd.read_csv(csv)#
        print(  len(df)    )
        print(  df.columns )
        print( df.head(5)  )
        new_name ={ 'F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv': 'touzizhidao',
                    'F:\\聊天机器人\\Q-A-matching\\data\\baoxian_zhidao\\baoxianzhidao_filter.csv':'baoxianzhidao'}
        process('F:\\聊天机器人\\%s.csv'%new_name,df)







    exit()
    
  
  