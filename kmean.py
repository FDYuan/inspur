# -*- coding: utf-8 -*-
# 中文分词、计算TF—IDF、Kmeans聚类
# 外部文件：sqlite数据库 data.db
# 表结构为 news (id interger,title varchar(255),content text,clicks interger,date text，type interger)
# 输入为数据库的title和content列
# 输出为更新数据库的type列
import sqlite3
import time
import codecs
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

databasename = 'data.db'

reload(sys)
sys.setdefaultencoding('utf-8')
def getlist():
    lines=[]
    with codecs.open('words.csv', 'r', 'utf-8') as f:
        for line in f:
            lines.append(str(line).replace(',',' '))
    return lines

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
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    mat=vectorizer.fit_transform(lines)
    tfidf = transformer.fit_transform(mat)
    names=vectorizer.get_feature_names()
    np.save('mat',mat)
    np.save('names',names)
    X = tfidf.toarray()
    return X,tfidf


# Kmeans聚类
def process(tfidf, k):
    km_cluster = KMeans(n_clusters=k, max_iter=100, n_init=10,
                        init='k-means++')
    # km_cluster = MiniBatchKMeans(n_clusters=10, max_iter=100, n_init=1,
    #                     init='k-means++',init_size=3000,batch_size=1000)
    result = km_cluster.fit_predict(tfidf)
    return result


def Draw(silhouette_avg, sample_silhouette_values, X, y, k):
    # 创建一个 subplot with 1-row 2-column
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)
    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.1, 1])
    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    ax1.set_ylim([0, len(X) + (k + 1)])
    y_lower = 0
    for i in range(k):  # 分别遍历这几个聚类
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.Spectral(float(i)/k)  # 搞一款颜色
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)  # 这个系数不知道干什么的
        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper
    # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")
    plt.show()

if __name__=='__main__':
    print u"Begin"
    lines=getlist()
    print u"求tf-idf..."
    X,tfidf = gettfidf(lines)
    print u'K-means...'
    for k in range(2,26):
        result = process(tfidf, k)
        print metrics.calinski_harabaz_score(X, result)
    #print u"更新分类..."
    #updatesql(result)
    print u'评估...'
    #silhouette_avg = metrics.silhouette_score(X, result)  # 平均轮廓系数
    #print silhouette_avg # 每个点的轮廓系数
    #sample_silhouette_values = metrics.silhouette_samples(X, result)
    #Draw(silhouette_avg, sample_silhouette_values, X, result, k)
