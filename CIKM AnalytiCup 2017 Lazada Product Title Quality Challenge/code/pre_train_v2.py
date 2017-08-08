import pandas as pd
import numpy as np
import nltk
#nltk.download('all')
#print(nltk.help.upenn_tagset())
pos_tags=[
        'CC',
        'CD',
        'DT',
        'EX',
        'FW',
        'IN',
        'JJ',
        'JJR',
        'JJS',
        'LS',
        'MD',
        'NN',
        'NNS',
        'NNP',
        'NNPS',
        'PDT',
        'POS',
        'PRP',
        'PRP$',
        'RB',
        'RBR',
        'RBS',
        'RP',
        'SYM',
        'TO',
        'UH',
        'VB',
        'VBD',
        'VBG',
        'VBN',
        'VBP',
        'VBZ',
        'WDT',
        'WP',
        'WP$',
        'WRB'
]
from bs4 import BeautifulSoup
pos_datas={}

train=pd.read_csv('../training/data_train.csv',encoding = 'utf-8')
clarity=pd.read_csv('../training/clarity_train.csv')
conciseness=pd.read_csv('../training/conciseness_train.csv')
description=train["short_description"].values
title=train["title"].values

for tag in pos_tags:
    pos_datas[tag]=[]
    pos_datas["RATE_"+tag]=[]
pos_datas["LEN"]=[]
pos_datas["OLEN"]=[]
for tag in pos_tags:
    pos_datas["TITLE_"+tag]=[]
    pos_datas["TITLE_RATE_"+tag]=[]
pos_datas["TITLE_LEN"]=[]
pos_datas["TITLE_OLEN"]=[]
for i,htmltext in enumerate(description):
    pos_data={}
    pos_len=0
    if(not isinstance(htmltext, str)):
        for tag in pos_tags:
            pos_data[tag]=0
        
        for tag in pos_tags:
            pos_datas[tag].append(pos_data[tag])
            pos_datas["RATE_"+tag].append(pos_data[tag])
        pos_datas["LEN"].append(pos_len)
        pos_datas["OLEN"].append(pos_len)
        continue
    cleantext=BeautifulSoup(htmltext, 'html.parser').text
    cleantext=nltk.word_tokenize(cleantext)
    cleantext=nltk.pos_tag(cleantext)
    for tag in pos_tags:
        pos_data[tag]=0
    for value in cleantext:
        tag=value[1]
        if(tag in pos_tags):
            pos_data[tag]+=1
            pos_len+=1
    for tag in pos_tags:
        pos_datas[tag].append(pos_data[tag])
        if(pos_len==0):
            rate=pos_data[tag]/1
        else:
            rate=pos_data[tag]/pos_len
        pos_datas["RATE_"+tag].append(rate)
    pos_datas["LEN"].append(pos_len)
    pos_datas["OLEN"].append(len(cleantext))
for i,htmltext in enumerate(title):
    pos_data={}
    pos_len=0
    if(not isinstance(htmltext, str)):
        for tag in pos_tags:
            pos_data["TITLE_"+tag]=0
        
        for tag in pos_tags:
            pos_datas["TITLE_"+tag].append(pos_data["TITLE_"+tag])
            pos_datas["TITLE_RATE_"+tag].append(pos_data["TITLE_RATE_"+tag])
        pos_datas["TITLE_LEN"].append(pos_len)
        pos_datas["TITLE_OLEN"].append(pos_len)
        continue
    cleantext=BeautifulSoup(htmltext, 'html.parser').text
    cleantext=nltk.word_tokenize(cleantext)
    cleantext=nltk.pos_tag(cleantext)
    for tag in pos_tags:
        pos_data["TITLE_"+tag]=0
    for value in cleantext:
        tag=value[1]
        if(tag in pos_tags):
            pos_data["TITLE_"+tag]+=1
            pos_len+=1
    for tag in pos_tags:
        pos_datas["TITLE_"+tag].append(pos_data["TITLE_"+tag])
        if(pos_len==0):
            rate=pos_data["TITLE_"+tag]/1
        else:
            rate=pos_data["TITLE_"+tag]/pos_len
        pos_datas["TITLE_RATE_"+tag].append(rate)
    pos_datas["TITLE_LEN"].append(pos_len)  
    pos_datas["TITLE_OLEN"].append(len(cleantext))  

  
del train["short_description"]
del train["title"]
for tag in pos_tags:
    train[tag]=pos_datas[tag]
    train["RATE_"+tag]=pos_datas["RATE_"+tag]
for tag in pos_tags:
    train["TITLE_"+tag]=pos_datas["TITLE_"+tag]
    train["TITLE_RATE_"+tag]=pos_datas["TITLE_RATE_"+tag]
train["LEN"]=pos_datas["LEN"]
train["OLEN"]=pos_datas["OLEN"]
train["TITLE_LEN"]=pos_datas["TITLE_LEN"]
train["TITLE_OLEN"]=pos_datas["TITLE_OLEN"]
train["clarity"]=clarity
train["conciseness"]=conciseness
train.to_csv('../training/data_train_pre_v2.csv',index=False)
