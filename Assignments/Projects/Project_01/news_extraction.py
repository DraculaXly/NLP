'''
# First edition at 2019.11.1 by Dracula
Input: string
Output: list of everyone's points
'''

# Import all the libs
import re
import jieba
import gensim
import numpy as np
import synonyms

from gensim import models
from functools import wraps
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from pyltp import Parser
from pyltp import SementicRoleLabeller

# path of pyltp model
cws_model = r'D:\GitHub\Pyltp\ltp_data\cws.model'
pos_model = r'D:\GitHub\Pyltp\ltp_data\pos.model'
ner_model = r'D:\GitHub\Pyltp\ltp_data\ner.model'
par_model = r'D:\GitHub\Pyltp\ltp_data\parser.model'
srl_model = r'D:\GitHub\Pyltp\ltp_data\srl\pisrl_win.model'

# Useful link
# https://ltp.readthedocs.io/zh_CN/latest/appendix.html

# Initial pyltp
# 分词
segmentor = Segmentor()
# 词性标注
postagger = Postagger()
# 命名实体识别
recognizer = NamedEntityRecognizer()
# 依存句法分析
parser = Parser()
# 语义角色标注
labeller = SementicRoleLabeller()

# 用自己之前训练的词向量获取近似词
wiki_model = models.Word2Vec.load(r'D:\GitHub\Data\zhwiki\wiki_corpus.model')
def get_similar_words(word):
    visited = set(word)
    
    words = wiki_model.wv.most_similar(word, topn=10)
    words = [w[0] for w in words]
    
    for w in words:
        ext_words = wiki_model.wv.most_similar(w, topn=10)
        for w in ext_words:
            visited.add(w[0])
    
    return list(visited)

#similar_words_1 = get_similar_words('说')

# 尝试使用直接封装好的近义词工具包试试
# https://github.com/huyingxi/Synonyms/
def get_similar_words_2(word):
    visited = set(word)
    
    words = synonyms.nearby(word)[0]
    
    for w in words:
        ext_words = synonyms.nearby(w)[0]
        for w in ext_words:
            visited.add(w)
    
    return list(visited)
#similar_words_2 = get_similar_words_2('说')

# 多次获取直接获得一系列与说接近的词
say_words = [
            '诊断', '交代', '说', '说道', '指出','报道','报道说','称', '警告',
            '所说', '告诉', '声称', '表示', '时说', '地说', '却说', '问道', '写道', 
            '答道', '感叹', '谈到', '说出', '认为', '提到', '强调', '宣称', '表明', 
            '明确指出', '所言', '所述', '所称', '所指', '常说', '断言', '名言', '告知', 
            '询问', '知道', '得知', '质问', '问', '告诫', '坚称', '辩称', '否认', '还称', 
            '指责', '透露', '坦言', '表达', '中说', '中称', '他称', '地问', '地称', '地用',
            '地指', '脱口而出', '一脸', '直说', '说好', '反问', '责怪', '放过', '慨叹', '问起',
            '喊道', '写到', '如是说', '何况', '答', '叹道', '岂能', '感慨', '叹', '赞叹', '叹息',
            '自叹', '自言', '谈及', '谈起', '谈论', '特别强调', '提及', '坦白', '相信', '看来', 
            '觉得', '并不认为', '确信', '提过', '引用', '详细描述', '详述', '重申', '阐述', '阐释',
            '承认', '说明', '证实', '揭示', '自述', '直言', '深信', '断定', '获知', '知悉', '得悉', 
            '透漏', '追问', '明白', '知晓', '发觉', '察觉到', '察觉', '怒斥', '斥责', '痛斥', '指摘',
            '回答', '请问', '坚信', '一再强调', '矢口否认', '反指', '坦承', '指证', '供称', '驳斥', 
            '反驳', '指控', '澄清', '谴责', '批评', '抨击', '严厉批评', '诋毁', '责难', '忍不住', 
            '大骂', '痛骂', '问及', '阐明'
            ]

# 命名实体识别
def get_NamedEntityRecognizer(sentence):
    segmentor.load(cws_model)
    words = segmentor.segment(sentence) #分词
    
    postagger.load(pos_model)
    postags = postagger.postag(words) #词性标注
    
    recognizer.load(ner_model)
    netags = recognizer.recognize(words, postags) #命名实体识别
    segmentor.release()
    postagger.release()
    recognizer.release()
    return list(netags)

# 句子依存分析
def get_Parser(sentence):
    segmentor.load(cws_model)
    words = segmentor.segment(sentence) #分词
    
    postagger.load(pos_model)
    postags = postagger.postag(words) #词性标注
    
    parser.load(par_model)
    arcs = parser.parse(words, postags) #依存句法分析
    segmentor.release()
    postagger.release()
    parser.release()
    return arcs

# 获取主语
def get_name(name, predic, words, proper):
    '''
    name: name
    predic: verb
    words: result of cutting sentence
    proper: acrs.relation of the result of get_Parser()
    '''
    index = words.index(name)
    
    cut_proper = proper[index + 1:]
    pre = words[: index]
    pos = words[index + 1:]
    # 向前拼接主语的定语
    while pre:
        w = pre.pop(-1)
        w_index = words.index(w)
        
        if proper[w_index] == 'ADV': continue
        if proper[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，', '。', '、', '）', '（']):
            name += w
        else:
            pre = False
    
    # 向后拼接
    while pos:
        w = pos.pop(0)
        p = cut_proper.pop(0)
        if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
            name += w
        else:
            return name
    return name

# 获取谓语之后的句子
def get_saying(words, proper, heads, pos):
    '''
    words: result of cutting sentence
    proper: acrs.relation of the result of get_Parser()
    heads: list of heads of the result of get_Parser()
    pos: postion
    '''
    # 默认谓语动词后面是：的话就是说的话
    if '：' in words:
        return ''.join(words[words.index('：') + 1:])
    # 其它
    while pos < len(words):
        w = words[pos]
        p = proper[pos]
        h = heads[pos]
        
        # 谓语还未结束
        if p in ['DBL', 'CMP', 'RAD']:
            pos += 1
            continue
        # 定语
        if p == 'ATT' and proper[h-1] != 'SBV':
            pos = h
            continue
        # 宾语
        if p == 'VOB':
            pos += 1
            continue
        else:
            if w == "，":
                return ''.join(words[pos + 1:])
            else:
                return ''.join(words[pos:])

# 解析句子，获得主语和宾语，以列表形式输出
def parse_sentence(sentence):
    # 初始化
    segmentor.load(cws_model)
    cuts = list(segmentor.segment(sentence)) #分词
    
    # 检查是否有跟说有关的
    say_ws = [w for w in cuts if w in say_words]
    
    if not say_ws: return False
    wp = get_Parser(sentence) # 句子依存分析
    wp_relation = [w.relation for w in wp]
    postagger.load(pos_model)
    postags = list(postagger.postag(cuts)) #词性标注
    
    name = ''
    stack = []
    result = []
    # 单个依存分析
    for k, v in enumerate(wp):
        # 如果出现了主语 person name/organization name/geographical name
        if postags[k] in ['nh', 'ni', 'ns']:
            # 存储这些主语
            stack.append(cuts[k])
        # 确定第一个主语
        if v.relation == 'SBV' and (cuts[v.head - 1] in say_ws):
            name = get_name(cuts[k], cuts[v.head - 1], cuts, wp_relation)
            saying = get_saying(cuts, wp_relation, [i.head for i in wp], v.head)
            if not saying:
                quotations = re.findall(r'“(.+?)”', sentence)
                if quotations: saying = quotations[-1]
            return [name, saying]
        if cuts[k] == '：':
            name = stack.pop()
            saying = ''.join(cuts[k + 1:])
            return [name, saying]
    segmentor.release()
    postagger.release()
    return False

# 计算句向量
def sentence_vec(sentence):
    segmentor.load(cws_model)
    words = list(segmentor.segment(sentence))
    sent_vec = np.zeros(100)
    for w in words:
        try:
            if wiki_model.wv[w].any():
                sent_vec += wiki_model.wv[w]
        except KeyError:
            wiki_model.wv[w] = np.zeros(100)
            sent_vec += wiki_model.wv[w]
    sent_vec = sent_vec / len(words)
    return sent_vec

# 句子之间的相关性
def cosine_dis(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * (np.linalg.norm(vec2)))

# 将一段话拆分出来，并按照句子之间的相似性合并
def get_sentence(string):
    # 分句
    sentences = re.findall('.*?[.。?？!！]+', string)
    result = []
    
    while len(sentences) > 1: # 如果有不止一个句子
        s1 = sentence_vec(sentences[0])
        s2 = sentence_vec(sentences[1])
        # 如果不相似，则弹出第一个，重新判断接下来的
        i = cosine_dis(s1, s2)
        if i < 0.5:
            a = sentences.pop(0)
            result.append(a)
        else:
            # 有相关性的话，就将两个相加，并从列表中剔除一个
            sentences[0] = sentences[0] + sentences[1]
            sentences.pop(1)
    result.append(sentences[0])
    return result

# 整合功能，输入一段话，输出每个人说话的内容
def show_ans(string):
    sents = get_sentence(string)
    news_ext = []
    for w in sents:
        result = parse_sentence(w)
        news_ext.append(result)
    return news_ext

if __name__ == "__main__":
    with open('graph.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        ans = show_ans(text)
        print(ans)