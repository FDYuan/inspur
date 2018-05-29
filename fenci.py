# -*- coding: utf-8 -*-
# 中文分词、计算TF—IDF、Kmeans聚类
# 外部文件：sqlite数据库 data.db
# 表结构为 news (id interger,title varchar(255),content text,clicks interger,date text，type interger)
# 输入为数据库的title和content列
# 输出为更新数据库的type列
import csv
import json
import re
import sqlite3
import sys

import jieba
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

reload(sys)
sys.setdefaultencoding('utf-8')

databasename = 'data.db'
stopwords = ''


# 数据库读数据 从sqlite数据库
def getdata(databasename):
    conn = sqlite3.connect(databasename)
    conn.text_factory = str
    cursor = conn.cursor()
    cursor.execute('select title,content from news')
    re = cursor.fetchall()
    cursor.close()
    conn.commit()
    conn.close()
    return re


# 每个新闻取前100个词进行中文分词 输出为list[list]
def fenci(rows):
    dr = re.compile(r'</?\w+[^>]*>')
    line = re.compile(r'\n+')
    dig = re.compile(r'[0-9]|[A-Z]|[a-z]')
    result = []
    for row in rows:
        text = dig.sub('', line.sub('\n', dr.sub('', row[0])+dr.sub('', row[1]))).replace('&nbsp;', '').replace('&gt;', '>').replace(
            '&lt;', '<').replace('\t', '').replace('&amp;', '&').replace('&quot;', '"').decode('utf8', 'ignore')[0:100].encode('utf8')
        words = delstop(text, stopwords)
        result.append(words)
        # result.append(jieba.analyse.extract_tags(text,20))
    return result


# 获取停止词 输出位set
def stopword(stopword):
    with open(stopword, "r") as fp:
        words = fp.read()
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)


# 去除停止词 输出为list
def delstop(words, stopset):
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stopset:
            new_words.append(r)
    return new_words


# 更新数据库
def updatesql(result):
    conn = sqlite3.connect(databasename)
    conn.text_factory = str
    cursor = conn.cursor()
    for i in range(len(result)):
        cursor.execute('UPDATE news SET type =%s WHERE id=%s' %
                       (result[i], i+1))
    cursor.close()
    conn.commit()
    conn.close()


# 计算TF-IDF
def gettfidf(lines):
    text = []
    for line in lines:
        words = ""
        for word in line:
            words += word+" "
        text.append(words)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
    return tfidf


# Kmeans聚类
def process(tfidf):
    km_cluster = KMeans(n_clusters=10, max_iter=100, n_init=10,
                        init='k-means++', n_jobs=6)
    result = km_cluster.fit_predict(tfidf)
    return result


if __name__ == '__main__':
    print "读取数据..."
    rows = getdata(databasename)
    stopwords = stopword('stopword.txt')
    print "分词..."
    lines = fenci(rows)
    print "求tf-idf..."
    tfidf = gettfidf(lines)
    print 'K-means...'
    result = process(tfidf)
    print "更新分类..."
    updatesql(result)
