"""
Fisrt edition: 2019.10.28 by Dracula
Use the news to train model
"""

# Import all lib
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  #show chinese
import networkx
import random
import re

from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim import models
from functools import reduce
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from json import dumps

# split_sentence
def split_sentence(string):
    str1 = string.replace('\r\n', '').strip()
    sentence = re.split(r'[.。?？!！]', str1)
    pure_sentence = [i for i in sentence if i]
    return pure_sentence

# cut sentence to words
def cut_sentence_to_words(sentence):
    sentence = re.sub(r'[^0-9A-Za-z\u4e00-\u9fa5]', '', sentence)
    return [w for w in jieba.cut(sentence)]

# get the sentence-words list
def get_sentence_list(sentences):
    sentence_list = []
    for sent in sentences:
        words = cut_sentence_to_words(sent)
        if len(words) > 0:
            sentence_list.append(words)
    return sentence_list

# Do sentence transform to words
def transform(string):
    sentences = split_sentence(string)
    sentence_words = get_sentence_list(sentences)
    return sentence_words

# use the news to train word vector
fpath = r'D:\GitHub\Data\sqlResult_1558435.csv'

news_content = pd.read_csv(fpath, encoding='gb18030')

df = pd.DataFrame()
df['content'] = news_content['content']
df = df.fillna('')
df['tokenized_content'] = df['content'].apply(transform)

# write each sentence-words
with open('all_corpus.txt', 'w', encoding='utf-8') as f:
    for i in range(len(df)):
        for e in df['tokenized_content'][i]:
            f.write(' '.join(e))
            f.write('\n')

# train
model = FastText(LineSentence('all_corpus.txt'), window=5, size=40, iter=10, min_count=1, workers=8)
model.save('Summarization.model')

# store the frequency
tokeners = [w for l in df['tokenized_content'] for t in l for w in t]
tokener_length = len(tokeners)
tokener_counter = Counter(tokeners)
frequency = {w : count / tokener_length for w, count in tokener_counter.items()}
freq_file = dumps(frequency)
with open('frequency.json', 'w', encoding='utf-8') as f:
    f.write(freq_file)