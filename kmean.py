# -*- coding: utf-8 -*-
# 中文分词、计算TF—IDF、Kmeans聚类
# 外部文件：sqlite数据库 data.db
# 表结构为 news (id interger,title varchar(255),content text,clicks interger,date text，type interger)
# 输入为数据库的title和content列
# 输出为更新数据库的type列
import codecs
import sqlite3
import sys
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import (AgglomerativeClustering, Birch, KMeans,
                             DBSCAN,MiniBatchKMeans)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA

databasename = 'data.db'

reload(sys)
sys.setdefaultencoding('utf-8')


def getlist():
    lines = []
    with codecs.open('cut.csv', 'r', 'utf-8') as f:
        for line in f:
            lines.append(str(line).replace(',', ' '))
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
    mat = vectorizer.fit_transform(lines)
    tfidf = transformer.fit_transform(mat)
    names = vectorizer.get_feature_names()
    np.save('mat', mat)
    np.save('names', names)
    return tfidf


def PCA(tfidf, dimension):
    weight = tfidf.toarray()

    print u'原有维度: ', len(weight[0])
    pca = PCA(n_components=dimension)  # 初始化PCA
    X = pca.fit_transform(weight)  # 返回降维后的数据
    print u'降维后维度: ', len(X[0])
    return X

# Kmeans聚类


def kmeans(tfidf, k):
    clusterer = KMeans(n_clusters=k,init='k-means++')
    # clusterer = AgglomerativeClustering(n_clusters=k)
    y = clusterer.fit_predict(tfidf)
    return y,clusterer


def birch(X, k):
    clusterer = DBSCAN(n_jobs=-1)
    y = clusterer.fit_predict(X)
    print y
    return y


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


if __name__ == '__main__':
    k=19
    print u"Begin"
    lines = getlist()
    print u"求tf-idf..."
    tfidf = gettfidf(lines)
    X = tfidf.toarray()
    print u'K-means...'
    #for k in range(2, K):
    start = time.time()
    y,clu = kmeans(tfidf, k)
    stop=time.time()
    print u'聚类时间为：'+str(stop- start)
    cen=clu.cluster_centers_
    np.save('center',cen)
    # print metrics.calinski_harabaz_score(X, y)
    # point.append(sum(np.min(cdist(X, result.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    # y = birch(X, k)
    # silhouette_avg = metrics.silhouette_score(X, y)  # 平均轮廓系数
    # print silhouette_avg
    # point.append(silhouette_avg)
    # sample_silhouette_values = metrics.silhouette_samples(X, y)  # 每个点的轮廓系数
    # Draw(silhouette_avg, sample_silhouette_values, X, y, k)
    #print y
    #print u"更新分类..."
    #updatesql(y)
