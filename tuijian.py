# -*- coding: UTF-8 -*-
# 将用户标签构造为向量 加入数据集进行聚类 同类即为候选集
# 同类中利用轮廓系数作
# 排名算式为 K=a*轮廓系数+b*时间+c*点击率 (a,b,c)初步定为(0.7,0.2,0.1)
# 传入稀疏矩阵 mat list(names)
import datetime
import sqlite3
import time

import jieba.analyse
import numpy as np
import scipy.stats
from numpy import linalg
from scipy.sparse import csr

# from fenci import FenCi

input_text = u"济南"
databasename = 'data.db'


# 读取新闻的点击量和日期信息
def getdata(databasename):
    conn = sqlite3.connect(databasename)
    conn.text_factory = str
    cursor = conn.cursor()
    cursor.execute('select clicks,date from news')
    re = cursor.fetchall()
    cursor.close()
    conn.commit()
    conn.close()
    return re


# 计算点击量和日期的权值
def getweight(data):
    clickw = []
    timew = []
    now = time.time()
    for pen in data:
        click = pen[0]
        cw = click/8000.0
        clickw.append(cw)
        date = pen[1]
        unix = time.mktime(time.strptime(date[:19], '%Y-%m-%d %H:%M:%S'))
        tw = scipy.stats.norm(0, 0.4).cdf(((now-unix)/172800)/2)*2-1.0
        timew.append(tw)
    return clickw, timew


# 计算两个变量的夹角
def cosvector(vec1, vec2):
    num = float(vec1.T * vec2)  # 若为行向量则 A * B.T
    denom = linalg.norm(vec1) * linalg.norm(vec2)
    cos = num / denom  # 余弦值
    return cos
def coscos(vec1,vec2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vec1, vec2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

# 计算用户的特征变量
def getvector(names, words):
    l = len(names)
    vec = np.zeros(l)
    for word in words:
        for i in range(l):
            if word==names[i]:
                vec[i]+=1
    return vec


# 计算夹角的算式
def getcos(mat, vec):
    mat=mat.tolist()
    # arr=mat.toarray()
    re = np.zeros(mat.shape[0])
    for i in range(mat.shape[1]):
        re[i] = cosvector(mat.getrow(i).toarray(), vec)
    return re


# 推荐K值
def getK(cos, clicks, times):
    re = 0.7*cos+0.2*clicks+0.1*times
    return re


# 分词
# ob = FenCi()
# stopwords = ob.stopword('stopword.txt')
jieba.analyse.set_stop_words('stopword.txt')
lines = jieba.analyse.extract_tags(input_text)
# 计算向量夹角
mat = np.load('mat.npy')
names = np.load('names.npy')
print '向量计算...'
vec = getvector(names, lines)
cos = getcos(mat, vec)
# 权值计算
data = getdata(databasename)
clicks, times = getweight(data)
weight = getK(cos, clicks, times)
print weight.sort()
