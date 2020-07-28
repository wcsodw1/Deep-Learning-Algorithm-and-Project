#!/usr/bin/env python
# coding: utf-8

# # data preprocess and words cut
# 
# - #### 以PTT推噓文為例 做資料前處理

# In[1]:


import pandas as pd
import re
import os
import jieba # 專門用來處理中文斷詞的系統
import jieba.posseg as pseg 
import pickle
import numpy as np


# In[2]:


## Find the path of this file

# 如果檔案需從外部進入可以這樣用 (turn back to main directory)
# os.chdir("../")
# article = pd.read_csv('Part1/Data/article_practice.csv')

import os
os.chdir("../")
os.getcwd()


# ## data overview

# In[3]:


# load data (Set Path)
article = pd.read_csv('Data/article_practice.csv')


# In[4]:


# Check Data
article.head()


# In[5]:


article['content']


# In[7]:


article['content'] = article['content'].str.replace('https?:\/\/\S*', '')
article['content']

# ex2: 將 ''(空值) 取代為'NaN'
article['content'] = article['content'].replace('', np.NaN)

article['idx'] = article.index
article['idx']


# In[8]:


# check type
type(article['content'])

# 1.Preprocessing  :  filter rules(依據Domain Knowledge去除不需要的資訊)
   
# 1.1 ex1: 將'https?:\/\/\S*' 取代為' '(空值)
article['content'] = article['content'].str.replace('https?:\/\/\S*', '')

# 1.2 將 ''(空值) 取代為'NaN'
article['content'] = article['content'].replace('', np.NaN)

article['idx'] = article.index
article['idx']

# 1.3 
article = article.dropna()
article


# In[8]:


article['content']


# In[9]:


article['idx']


# In[10]:


# remove data

#remove NaN
article = article.dropna()

#讓index重置成原本的樣子
article = article.reset_index(drop=True)

article['idx'] = article.index
article['idx']


# String Split 
# ex: 依照文字之間的空格split

article['content'].str.split( " ", n=4, expand = True )


# In[7]:


article.to_csv('data/article_preprocessed.csv', index=False)


# # jieba

# ## cut word : 自然語言處理第一招 : 處理斷詞 
# 
# - ### 斷詞是甚麼? 中文詞句的斷句!

# In[8]:


## set dictionary (can define yourself)

# jieba段詞的dictionary(有很多種 這邊只是某個範例)
jieba.set_dictionary('jieba/dict.txt.big')

#斷詞檔案 :  jieba為中文斷詞的資料庫 
stop_words = open('jieba/stop_words.txt',encoding="utf-8").read().splitlines()


# In[9]:


print(stop_words[:300])


# In[10]:


data = pd.read_csv('data/article_preprocessed.csv')
data = data['content'].tolist()


# In[11]:


print(data[:5])


# In[12]:


sentences = []

for i, text in enumerate(data):
    line = []

    for w in jieba.cut(text, cut_all=False):
        
        ## remove stopwords and digits
        ## can define your own rules
        if w not in stop_words and not bool(re.match('[0-9]+', w)):
            line.append(w)

    sentences.append(line)

    if i%10000==0:
        print(i, '/', len(data))


# In[13]:


print(sentences[0:20])


# In[14]:


## save data as pickle format
with open("data/article_cutted", "wb") as file : 
    pickle.dump(sentences, file)


# - ## posseg (詞性)
# 
#  #### 在斷詞時 , 請用 jieba 把詞性抓出來 (1 筆即可)

# In[15]:


import jieba.posseg as pseg


# In[16]:


for w, f in pseg.cut(data[0]):
    print(w, ' ', f)

