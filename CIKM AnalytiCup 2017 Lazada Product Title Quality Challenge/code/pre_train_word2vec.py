import pandas as pd
import numpy as np
import nltk
from bs4 import BeautifulSoup
from unidecode import unidecode
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import antispam
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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
count_tags=['li','br','img','a']
pos_datas={}
for tag in pos_tags:
    pos_datas[tag]=[]
for tag in count_tags:
    pos_datas[tag]=[]
pos_datas["LEN"]=[]
pos_datas["OLEN"]=[]
for tag in pos_tags:
    pos_datas["TITLE_"+tag]=[]
pos_datas["TITLE_LEN"]=[]
pos_datas["TITLE_OLEN"]=[]
pos_datas["IS_HTML"]=[]

#antispam
pos_datas["title_antispam"]=[]
pos_datas["description_antispam"]=[]


# word2vec
pos_datas["category_title_w2c"]=[]
pos_datas["category_title_avg_w2c"]=[]
pos_datas["category_title_avg_related"]=[]
pos_datas["category_title_vg_related"]=[] #>0.46 ~ very good
pos_datas["category_title_g_related"]=[] #>0.3 ~ good
pos_datas["category_title_s_related"]=[] #>0.14 ~ soso
pos_datas["category_description_w2c"]=[]
pos_datas["category_description_avg_related"]=[]
pos_datas["category_description_avg_w2c"]=[]
pos_datas["category_description_vg_related"]=[] #>0.46 ~ very good
pos_datas["category_description_g_related"]=[] #>0.3 ~ good
pos_datas["category_description_s_related"]=[] #>0.14 ~ soso

train=pd.read_csv('../training/data_train.csv',encoding = 'utf-8')
clarity=pd.read_csv('../training/clarity_train.csv')
conciseness=pd.read_csv('../training/conciseness_train.csv')
description=train["short_description"].values
categories=train["category_lvl_1"].values
title=train["title"].values
for i,htmltext in enumerate(description):
    try:
        htmltext=htmltext.lower()
    except:
        htmltext=htmltext
    category=categories[i].lower().split(" ")[0]
    pos_data={}
    pos_len=0
    if(not isinstance(htmltext, str)):
        for tag in pos_tags:
            pos_data[tag]=0
        
        for tag in pos_tags:
            pos_datas[tag].append(pos_data[tag])
        pos_datas["LEN"].append(pos_len)
        pos_datas["OLEN"].append(pos_len)
        pos_datas["IS_HTML"].append(0)
        for tag in count_tags:
            pos_data[tag]=0
        for tag in count_tags:
            pos_datas[tag].append(pos_data[tag])
        
        #antispam
        pos_datas["description_antispam"].append(1)
            
        #word2vec
        scores=[]
        vg=0
        g=0
        s=0
        words=[]
        for word in words:
            try:
                score=model.wv.similarity(category,word)
                if(score>=0.46):
                    vg+=1
                elif(score>=0.3):
                    g+=1
                elif(score>=0.14):
                    s+=1
                scores.append(score)
            except:
                pass
        if(len(scores)!=0):
            score=round(sum(scores),5)
            score_avg=round(sum(scores)/len(scores),5)
        else:
            score=0
            score_avg=0
        pos_datas["category_description_w2c"].append(score)
        pos_datas["category_description_avg_w2c"].append(score_avg)
        pos_datas["category_description_vg_related"].append(vg)
        pos_datas["category_description_g_related"].append(g)
        pos_datas["category_description_s_related"].append(s)    
        pos_datas["category_description_avg_related"].append(0)    
            
            
        continue
    html_tags=BeautifulSoup(htmltext,'html5lib').find_all()
    for tag in count_tags:
        pos_data[tag]=0
    if(len(html_tags)>0):
        pos_datas["IS_HTML"].append(1)
    else:
        pos_datas["IS_HTML"].append(0)
        
    for tag in html_tags:
        if(tag.name in count_tags):
            pos_data[tag.name]+=1
    for tag in count_tags:
        pos_datas[tag].append(pos_data[tag])
    cleantext=BeautifulSoup(htmltext, 'html.parser').text
    
    #antispam
    try:
        pos_datas["description_antispam"].append(antispam.score(cleantext))
    except:
        pos_datas["description_antispam"].append(0)
    words=nltk.word_tokenize(cleantext)
    cleantext=nltk.pos_tag(words)
    for tag in pos_tags:
        pos_data[tag]=0
    for value in cleantext:
        tag=value[1]
        if(tag in pos_tags):
            pos_data[tag]+=1
            pos_len+=1
    for tag in pos_tags:
        pos_datas[tag].append(pos_data[tag])
    pos_datas["LEN"].append(pos_len)
    pos_datas["OLEN"].append(len(cleantext))
    
    #word2vec
    scores=[]
    related=[]
    vg=0
    g=0
    s=0
    for k,word in enumerate(words):
        try:
            score=model.wv.similarity(category,word)
            if(score>=0.46):
                vg+=1
            elif(score>=0.3):
                g+=1
            elif(score>=0.14):
                s+=1
            scores.append(score)
            related.append(model.wv.similarity(words[k],words[k+1]))
        except:
            pass
    if(len(scores)!=0):
        score=round(sum(scores),5)
        score_avg=round(sum(scores)/len(scores),5)
    else:
        score=0
        score_avg=0
    if(len(related)!=0):
        avg_related=round(sum(related)/len(related),5)
    else:
        avg_related=0
    pos_datas["category_description_w2c"].append(score)
    pos_datas["category_description_avg_w2c"].append(score_avg)
    pos_datas["category_description_vg_related"].append(vg)
    pos_datas["category_description_g_related"].append(g)
    pos_datas["category_description_s_related"].append(s)
    pos_datas["category_description_avg_related"].append(avg_related)    

for i,htmltext in enumerate(title):
    try:
        htmltext=htmltext.lower()
    except:
        htmltext=htmltext
    category=categories[i].lower().split(" ")[0]
    pos_data={}
    pos_len=0
    if(not isinstance(htmltext, str)):
        for tag in pos_tags:
            pos_data["TITLE_"+tag]=0
        
        for tag in pos_tags:
            pos_datas["TITLE_"+tag].append(pos_data["TITLE_"+tag])
        pos_datas["TITLE_LEN"].append(pos_len)
        pos_datas["TITLE_OLEN"].append(pos_len)
        
        #antispam
        pos_datas["title_antispam"].append(1)
        
        #word2vec
        scores=[]
        vg=0
        g=0
        s=0
        words=[]
        for word in words:
            try:
                score=model.wv.similarity(category,word)
                if(score>=0.46):
                    vg+=1
                elif(score>=0.3):
                    g+=1
                elif(score>=0.14):
                    s+=1
                scores.append(score)
            except:
                pass
        if(len(scores)!=0):
            score=round(sum(scores),5)
            score_avg=round(sum(scores)/len(scores),5)
        else:
            score=0
            score_avg=0
        pos_datas["category_title_w2c"].append(score)
        pos_datas["category_title_avg_w2c"].append(score_avg)
        pos_datas["category_title_vg_related"].append(vg)
        pos_datas["category_title_g_related"].append(g)
        pos_datas["category_title_s_related"].append(s)
        pos_datas["category_title_avg_related"].append(0)
        
        
        continue
    cleantext=BeautifulSoup(htmltext, 'html.parser').text
    
    #antispam
    try:
        pos_datas["title_antispam"].append(antispam.score(cleantext))
    except:
        pos_datas["title_antispam"].append(0)
    
    words=nltk.word_tokenize(cleantext)
    cleantext=nltk.pos_tag(words)
    for tag in pos_tags:
        pos_data["TITLE_"+tag]=0
    for value in cleantext:
        tag=value[1]
        if(tag in pos_tags):
            pos_data["TITLE_"+tag]+=1
            pos_len+=1
    for tag in pos_tags:
        pos_datas["TITLE_"+tag].append(pos_data["TITLE_"+tag])
    pos_datas["TITLE_LEN"].append(pos_len)  
    pos_datas["TITLE_OLEN"].append(len(cleantext))  
    
    #word2vec
    scores=[]
    related=[]
    vg=0
    g=0
    s=0
    for k,word in enumerate(words):
        try:
            score=model.wv.similarity(category,word)
            if(score>=0.46):
                vg+=1
            elif(score>=0.3):
                g+=1
            elif(score>=0.14):
                s+=1
            scores.append(score)
            related.append(model.wv.similarity(words[k],words[k+1]))
        except:
            pass
    if(len(scores)!=0):
        score=round(sum(scores),5)
        score_avg=round(sum(scores)/len(scores),5)
    else:
        score=0
        score_avg=0
    if(len(related)!=0):
        avg_related=round(sum(related)/len(related),5)
    else:
        avg_related=0
    pos_datas["category_title_w2c"].append(score)
    pos_datas["category_title_avg_w2c"].append(score_avg)
    pos_datas["category_title_vg_related"].append(vg)
    pos_datas["category_title_g_related"].append(g)
    pos_datas["category_title_s_related"].append(s)
    pos_datas["category_title_avg_related"].append(avg_related)
  
del train["short_description"]
del train["title"]
for tag in pos_tags:
    train[tag]=pos_datas[tag]
for tag in count_tags:
    train[tag]=pos_datas[tag]
for tag in pos_tags:
    train["TITLE_"+tag]=pos_datas["TITLE_"+tag]
train["LEN"]=pos_datas["LEN"]
train["OLEN"]=pos_datas["OLEN"]
train["TITLE_LEN"]=pos_datas["TITLE_LEN"]
train["TITLE_OLEN"]=pos_datas["TITLE_OLEN"]

#antispam
train["title_antispam"]=pos_datas["title_antispam"]
train["description_antispam"]=pos_datas["description_antispam"]


#word2vec
train["category_title_w2c"]=pos_datas["category_title_w2c"]
train["category_title_avg_w2c"]=pos_datas["category_title_avg_w2c"]
train["category_title_vg_related"]=pos_datas["category_title_vg_related"]
train["category_title_g_related"]=pos_datas["category_title_g_related"]
train["category_title_s_related"]=pos_datas["category_title_s_related"]
train["category_title_avg_related"]=pos_datas["category_title_avg_related"]

train["category_description_w2c"]=pos_datas["category_description_w2c"]
train["category_description_avg_w2c"]=pos_datas["category_description_avg_w2c"]
train["category_description_vg_related"]=pos_datas["category_description_vg_related"]
train["category_description_g_related"]=pos_datas["category_description_g_related"]
train["category_description_s_related"]=pos_datas["category_description_s_related"]
train["category_description_avg_related"]=pos_datas["category_description_avg_related"]



train["clarity"]=clarity
train["conciseness"]=conciseness
train.to_csv('../training/data_train_pre_word2vec.csv',index=False,encoding='utf-8')
