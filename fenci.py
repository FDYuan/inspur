import csv
import re
import sqlite3
import sys

import jieba

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
            '&lt;', '<').replace('\t', '').replace('&amp;', '&').replace('&quot;', '"').replace('•', '').replace('▼', '').decode('utf8', 'ignore')[0:100].encode('utf8')
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


def outfile(lines):
    with open('words.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(lines)


print "读取数据..."
rows = getdata(databasename)
stopwords = stopword('stopword.txt')
print "分词..."
lines = fenci(rows)
outfile(lines)
