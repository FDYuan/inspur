import csv
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas

def getwords(filename):
    fp=open(filename,'r')
    words = pandas.read_csv(fp,'utf-8')
    return words

def getvector(lines):
    word_set = dict()
    for line in lines:
        words=line.split(",")
        for word in words:
            print word
            word_set |=dict(word)
    print word_set
    word_set = list(word_set)
    vsm = []
    for line in words:
        temp = []
        for word in word_set:
            temp.append(line.count(word) * 1.0)
        vsm.append(temp)
    mat = np.array(vsm)
    print mat
    csum = [float(len(np.nonzero(mat[:, i])[0])) for i in range(mat.shape[1])]
    csum = np.array(csum)
    csum = mat.shape[0] / csum
    idf = np.log(csum)
    idf = np.diag(idf)
    for vec in mat:
        if vec.sum() == 0:
            vec = vec / 1
        else:
            vec = vec / (vec.sum())
    tfidf = np.dot(mat, idf)
    return tfidf
def gettfidf(words):
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    mat=vectorizer.fit_transform(words)
    tfidf=transformer.fit_transform(mat)
    word=vectorizer.get_feature_names()
    print word
    weight=tfidf.toarray()
    for i in range(len(weight)):
        for j in range(word):
            print word[j],weight[i][j]

words = getwords('words.csv')
print words
# gettfidf(words)
