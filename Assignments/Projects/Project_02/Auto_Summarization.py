"""
Fisrt edition: 2019.10.28 by Dracula
Given a news and put the auto_summarization result out
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
from json import loads

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

# load model
model = models.Word2Vec.load(r"D:\Python\NLP\Projects\Project_02\Summarization.model")

# load frequency
with open('frequency.json', 'r', encoding='utf-8') as f:
    frequency = loads(f.read())

# get sentence vector
def sentence_embedding(sentence):
    # weight = alpah/(alpah + p)
    # alpha is a parameter, 1e-3 ~ 1e-5
    alpha = 1e-4
    global frequency

    max_fre = max(frequency.values())

    sentence_vec = np.zeros_like(model.wv['测试'])

    words = cut_sentence_to_words(sentence)
    words = [w for w in words if w in model.wv]

    for w in words:
        weight = alpha / (alpha + frequency.get(w, max_fre))
        sentence_vec += weight * model.wv[w]

    sentence_vec /= len(words)

    # Skip the PCA
    return sentence_vec

# get sentence similarity
def sentence_sim(sentence_list):
    sim_mat = np.zeros([len(sentence_list), len(sentence_list)])

    for i in range(len(sentence_list)):
        for j in range(len(sentence_list)):
            if i != j:
                sim_mat[i][j] = cosine(sentence_embedding(sentence_list[i]).reshape(1, 40),
                                       sentence_embedding(sentence_list[j]).reshape(1, 40))
    return sim_mat

# use textrank to get the result
def get_summary(news):
    sentences = split_sentence(news)
    sent_sim = sentence_sim(sentences)
    nx_graph = networkx.from_numpy_array(sent_sim)
    scores = networkx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print("摘要如下：\n", ranked_sentences[0][1])

# test the function
if __name__ == "__main__":
    with open('news.txt', 'r', encoding='utf-8') as f:
        news = f.read()
        get_summary(news)