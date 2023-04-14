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
    df['title'] = df['title'].astype(str).apply(lambda x: x.replace('\r','').replace('\n',';')  )
    df['reply'] = df['reply'].astype(str).apply(lambda x: x.replace('\r','').replace('\n',';')  )
    for index, row in df.iterrows():
        key   = row['title'].strip()
        value = row['reply'].strip()

        if key in ['',' ','nan' ] or  value in ['',' ','nan' ]  or  'nan' in key or 'nan' in 'value':
            continue

        f.write( str(key).strip().replace(' ', '') + '\n'    )
        f.write( str(value).strip().replace(' ', '') + '\n'  )
        f.write('\n\n')
    f.close()

    return


def process_common():

    manual_df = pd.read_excel('F:\\聊天机器人\\问答知识库\\手动整理.xlsx')
    manual_df = manual_df.append( manual_df )
    print(  manual_df.head(10) )

    import os, json
    #df = pd.read_json('F:\\聊天机器人\\问答知识库\\常识\\archive\\baike_qa_train.json',orient = 'records',lines='orient',nrows=700000)

    df = pd.DataFrame()
    for line in open('F:\\聊天机器人\\问答知识库\\常识\\archive\\baike_qa_train.json'):
        js      = json.loads(line)
        tmp_df  = pd.json_normalize(js)
        df      = df.append(tmp_df)

    df['title'] = df['title'].apply(lambda x: x.replace('"','')).apply(lambda x: x.replace('\r\n',''))
    df['answer'] = df['answer'].apply(lambda x: x.replace('"', '')).apply(lambda x: x.replace('\r\n',''))
    #这里处理太粗糙了
    print('df', df.head(10))
    baike_df = df[['title', 'answer']].rename(columns={'answer': 'reply'})
    #print('answear::::',baike_df['answer'])
    baike_df.to_csv('F:\\聊天机器人\\baike_df.csv')

    ############################################常识
    df          = pd.read_csv('F:\\聊天机器人\\问答知识库\\常识\\common1.csv')
    print('a_str',df['a_str'].tolist() )
    def get_key_answear( x ):
        split_in =  x.split('\t')
        if len(split_in)>=2:
            return split_in[1]
        else:
            return split_in
    df['a_str'] = df['a_str'].apply(lambda x :   get_key_answear(x)   )
    print( df.head(10) )

    jisuanji_df = df[['q_str','a_str']].rename( columns={'q_str':'title', 'a_str': 'reply'} )
    print(jisuanji_df.head(10))
    jisuanji_df.to_csv('F:\\聊天机器人\\jisuanji_df.csv')
    ########################################### laws
    df = pd.DataFrame()

    path = 'F:\\聊天机器人\\问答知识库\\laws\\train'
    for file in os.listdir(path):
        tmp = json.loads(open(path + '\\' + file).read());
        df = pd.concat([df, pd.json_normalize(tmp)], axis=0, ignore_index=True)

    path = 'F:\\聊天机器人\\问答知识库\\laws\\test'
    for file in os.listdir(path):
        df = pd.concat([df, pd.json_normalize(json.loads(open(path + '\\' + file).read()))], axis=0, ignore_index=True)

    law_df = df[['question', 'answer']].rename(columns={'question': 'title', 'answer': 'reply'})
    law_df.to_csv('F:\\聊天机器人\\law_df.csv')

    ###medical
    ###########################################medical
    import json
    df = pd.DataFrame()
    for line in open('F:\\聊天机器人\\问答知识库\\medical\medical.json'):
        # print('line:',line)
        js = json.loads(line)
        tmp_df = pd.json_normalize(js)
        df = df.append(tmp_df)
    print('df', df.head(10))
    df.to_excel('./newdf1.xlsx')

    df_list = []
    # 是什么
    tmp_df = df[['name', 'desc']].rename(columns={'name': 'title', 'desc': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '是什么'
    df_list.append(tmp_df)
    # 如何阻止
    tmp_df = df[['name', 'prevent']].rename(columns={'name': 'title', 'prevent': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '怎么预防'
    df_list.append(tmp_df)
    # 花费 cost_money
    tmp_df = df[['name', 'cost_money']].rename(columns={'name': 'title', 'cost_money': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '治疗费用'
    df_list.append(tmp_df)
    # recommand_eat
    tmp_df = df[['name', 'recommand_eat']].rename(columns={'name': 'title', 'recommand_eat': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '推荐饮食'
    df_list.append(tmp_df)
    # cure_lasttime
    tmp_df = df[['name', 'cure_lasttime']].rename(columns={'name': 'title', 'cure_lasttime': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '治疗时长'
    df_list.append(tmp_df)
    # symbol
    df['symptom'] = df['symptom'].apply(lambda x: '、'.join(x))
    tmp_df = df[['name', 'symptom']].rename(columns={'name': 'title', 'symptom': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '有哪些症状'
    df_list.append(tmp_df)

    tmp_df = df[['name', 'symptom']].rename(columns={'symptom': 'title', 'name': 'reply'});
    tmp_df['title'] = '身体有' + tmp_df['title'] + '症状，可能是得了哪个冰'
    df_list.append(tmp_df)
    # cause
    tmp_df = df[['name', 'cause']].rename(columns={'name': 'title', 'cause': 'reply'});
    tmp_df['title'] = tmp_df['title'] + '得病原因可能是'
    df_list.append(tmp_df)

    # recommand_drug
    tmp_df = df[['name', 'recommand_drug']].rename(columns={'name': 'title', 'recommand_drug': 'reply'});
    tmp_df['title'] = tmp_df['title'] + ',推荐什么药或有什么药推荐'
    df_list.append(tmp_df)
    medical_df = pd.concat(df_list, axis=0)
    # lines = open('F:\\聊天机器人\\问答知识库\\medical\medical.json').readlines()
    # df    = pd.json_normalize( lines )
    print(medical_df.head(10))
    medical_df['reply'] = medical_df['reply'].astype(str).apply( lambda x: x.replace('[]','' ) )
    medical_df  = medical_df.dropna(axis=0, how='any')
    medical_df =  medical_df[  (medical_df['title']!='')&( medical_df['reply']!='' ) ]
    medical_df.to_excel('F:\\聊天机器人\\medical_df.xlsx')
    # df1 = pd.read_csv('F:\\聊天机器人\\问答知识库\\medical\\test_list.txt', sep='\t', header=None, names=['title', 'reply'])
    # df2 = pd.read_csv('F:\\聊天机器人\\问答知识库\\medical\\train_list.txt', sep='\t', header=None, names=['title', 'reply'])
    # df = df1.append(df)
    # medical_df = df1.append(df)
    process('F:\\聊天机器人\\medical_df.txt', medical_df)
    ############touzi
    touzibaoxian_df = pd.DataFrame()
    for csv in ['F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv',
                'F:\\聊天机器人\\Q-A-matching\\data\\baoxian_zhidao\\baoxianzhidao_filter.csv'
                ]:
        df = pd.read_csv(csv) ; df['is_best'] = df['is_best'].apply(lambda x: str(x) )
        df = df[ df['is_best']=='1' ][['title','reply']]#
        print(  len(df)    )
        print(  df.columns )
        print(  df.head(5)  )
        new_name ={ 'F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv': 'touzizhidao',
                    'F:\\聊天机器人\\Q-A-matching\\data\\baoxian_zhidao\\baoxianzhidao_filter.csv':'baoxianzhidao'}
        #process('F:\\聊天机器人\\%s.csv'%new_name,df)
        touzibaoxian_df = touzibaoxian_df.append( df )
    touzibaoxian_df.to_csv('F:\\聊天机器人\\touzibaoxian_df.csv')

    big_common_df = pd.concat( [manual_df, baike_df.sample(frac=0.3),jisuanji_df,law_df, medical_df ,touzibaoxian_df ], axis=0 )

    big_common_df['title'] =  big_common_df['title'].astype(str).apply(lambda x: x.replace('\t','').replace('\n','').replace('\r','') )
    big_common_df['reply'] =  big_common_df['reply'].astype(str).apply(lambda x: x.replace('\t','').replace('\n','').replace('\r','') )
    big_common_df.to_csv('F:\\聊天机器人\\big_common.csv')

    big_common_df['remove'] = big_common_df['title'].apply(lambda x : 1 if '*' in  x  or x=='nan' else 0)
    big_common_df           = big_common_df[ big_common_df['remove']!=1 ]
    big_common_df           = big_common_df.dropna(axis=0, how='any')
    big_common_df = big_common_df[(big_common_df['title'] != '') & (big_common_df['reply'] != '')]
    process('F:\\聊天机器人\\big_common.txt', big_common_df)

    return


import  pandas as pd
if __name__ == '__main__':
    process_common(  )
    ###########################################心理咨询、尝试


    exit()


    process('F:\\聊天机器人\\%s.csv' % 'law', tmp_df)
    exit()

    #df = pd.read_csv('F:\\聊天机器人\\Q-A-matching\\data\\touzi_zhidao\\touzizhidao_filter.csv')

    process('F:\\聊天机器人\\%s.csv' % 'medical', df)
    exit()

    ########################################### touzi  baoxian
    exit()
    
  
  