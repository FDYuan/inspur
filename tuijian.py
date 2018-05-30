# 将用户标签构造为向量 加入数据集进行聚类 同类即为候选集
# 同类中利用轮廓系数作
# 排名算式为 K=a*轮廓系数+b*时间+c*点击率 (a,b,c)初步定为(0.7,0.2,0.1)
# 传入稀疏矩阵 mat list(names)
import datetime
import sqlite3
import time
from fenci import FenCi

import jieba
import numpy as np
import scipy.stats
from numpy import linalg

input_text = u"济南"
databasename = 'data.db'


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


def getweight(data):
    clickw = []
    timew = []
    now = time.time()
    for pen in data:
        click = pen[0]
        cw = 0.2*(click/8000.0)
        clickw.append(cw)
        date = pen[1]
        unix = time.mktime(time.strptime(date[:19], '%Y-%m-%d %H:%M:%S'))
        tw = 0.1*(scipy.stats.norm(0, 0.4).cdf(((now-unix)/172800)/2)*2-1.0)
        timew.append(tw)
    return clickw, timew


def cosvector(vec1, vec2):
    num = float(vec1.T * vec2)  # 若为行向量则 A * B.T
    denom = linalg.norm(vec1) * linalg.norm(vec2)
    cos = num / denom  # 余弦值
    return cos


def getvector(names, words):
    l = len(names)
    vec = np.zeros(l)
    for word in words:
        vec[names.argmax(word)] += 1
    return vec


# 分词
ob = FenCi()
stopwords = ob.stopword('stopword.txt')
lines = ob.fenci(input_text, stopwords)
# 计算向量夹角
mat = np.load('mat.npy')
names = np.load('names.npy')
vec = getvector(names, lines)
# 权值计算
data = getdata(databasename)
clicks, times = getweight(data)
