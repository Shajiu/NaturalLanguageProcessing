# -*- coding: utf-8 -*-
# @Time    : 2020/10/3 0:06
# @Author  : Shajiu
# @FileName: train.py
# @Software: PyCharm
# @Github  ：https://github.com/Shajiu
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn import svm

def read_csv(file):
    '''
    读取数据
    :param file: 文件路径
    :return: 数据
    '''
    return pd.read_csv(file,sep=',',nrows=15000)

def count_vectors_ridgeclassifier(train):
    '''
    Count Vectors + RidgeClassifier
    :param train:
    :return:
    '''
    vectorizer=CountVectorizer(max_features=3000)
    # 需要将dtype转换为object为unicode字符串
    train_test=vectorizer.fit_transform(train['text'].values.astype('U'))

    clf=RidgeClassifier()
    clf.fit(train_test,train['label'].values)

    val_pred=clf.predict(train_test)
    print(f1_score(train['label'].values,val_pred,average='macro'))

def tf_idf_ridgeclassifier(train):
    '''
    TF-IDF + RidgeClassifier
    :param train:
    :return:
    '''
    tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=3000)
    train_test=tfidf.fit_transform(train['text'].values.astype('U'))

    clf=RidgeClassifier()
    clf.fit(train_test,train['label'].values)

    val_pred=clf.predict(train_test)
    print(f1_score(train['label'].values,val_pred,average='macro'))

def Parameter_regularization(train):
    '''
    正则化参数对模型的影响
    :param train:
    :return:
    '''
    sample=train
    n=int(2*len(sample)/3)
    tfidf=TfidfVectorizer(ngram_range=(2,3),max_features=2500)
    train_test=tfidf.fit_transform(sample['text'].values.astype("U"))
    train_x=train_test[:n]
    train_y=sample['label'].values[:n]

    test_x=train_test[n:]
    test_y=sample['label'].values[n:]
    f1=[]
    for i in range(10):
        clf=RidgeClassifier(alpha=0.15*(i+1),solver='sag')
        clf.fit(train_x,train_y)
        val_pred=clf.predict(test_x)
        f1.append(f1_score(test_y,val_pred,average='macro'))

    plt.plot([0.15*(i+1) for i in range(10)],f1)
    plt.xlabel('alpha')
    plt.ylabel('f1_score')
    print(f1)
    plt.show()

def Max_Features(train):
    sample = train
    n = int(2 * len(sample) / 3)
    f1=[]
    features=[1000,2000,3000,4000]
    for i in range(4):
        tfidf=TfidfVectorizer(ngram_range=(2,3),max_features=features[i])
        train_test=tfidf.fit_transform(sample['text'].values.astype('U'))
        train_x=train_test[:n]
        train_y=sample['label'].values[:n]
        test_x=train_test[n:]
        test_y=sample['label'].values[n:]

        clf=RidgeClassifier(alpha=0.1*(i+1),solver='sag')
        clf.fit(train_x,train_y)
        val_pred=clf.predict(test_x)
        f1.append(f1_score(test_y,val_pred,average='macro'))
    plt.plot(features,f1)
    plt.xlabel('max_features')
    plt.ylabel('f1_score')
    plt.show()

def Ngram_Range(train):
    '''
    n-gram提取词语字符数的下边界和上边界，考虑到中文的用词习惯，ngram_range可以在(1,4)之间选取
    :param train:
    :return:
    '''
    sample=train
    n = int(2 * len(sample) / 3)
    f1=[]
    tfidf=TfidfVectorizer(ngram_range=(1,1),max_features=2000)
    train_test=tfidf.fit_transform(sample['text'].values.astype('U'))
    train_x=train_test[:n]
    train_y=sample['label'].values[:n]
    test_x=train_test[n:]
    test_y=sample['label'].values[n:]

    clf=RidgeClassifier(alpha=0.1*(1+1),solver='sag')
    clf.fit(train_x,train_y)
    val_pred=clf.predict(test_x)
    f1.append(f1_score(test_y,val_pred,average='macro'))

    tfidf = TfidfVectorizer(ngram_range=(2, 2), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'].values.astype("U"))
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1 * (2 + 1), solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))
    print(f1)

    tfidf = TfidfVectorizer(ngram_range=(3, 3), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'].values.astype('U'))
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1 * (3 + 1), solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))
    print(f1)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2000)
    train_test = tfidf.fit_transform(sample['text'].values.astype("U"))
    train_x = train_test[:n]
    train_y = sample['label'].values[:n]
    test_x = train_test[n:]
    test_y = sample['label'].values[n:]
    clf = RidgeClassifier(alpha=0.1 * (4 + 1), solver='sag')
    clf.fit(train_x, train_y)
    val_pred = clf.predict(test_x)
    f1.append(f1_score(test_y, val_pred, average='macro'))
    print(f1)

def LogisticRegression(train):
    '''
    LogisticRegression的目标函数
    :param train:
    :return:
    '''
    tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
    train_test=tfidf.fit_transform(train['text'].values.astype("U"))

    reg=linear_model.LogisticRegression(penalty='l2',C=1.0,solver='liblinear')
    reg.fit(train_test[:1000],train['label'].values[:1000])

    val_pred=reg.predict(train_test)
    print('预测结果中各类数目')
    print(pd.Series(val_pred).value_counts())
    print("\n F1 Score为：")
    print(f1_score(train['label'].values,val_pred,average='macro'))

def SGDClassifier(train):
    '''
    SGDClassifier使用mini-batch来做梯度下降，在处理大数据的情况下收敛更快
    :return:
    '''
    tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
    train_test=tfidf.fit_transform(train['text'].values.astype("U"))

    reg=linear_model.SGDClassifier(loss='log',penalty='l2',alpha=0.0001,l1_ratio=0.15)
    reg.fit(train_test,train['label'].values)

    val_pred=reg.predict(train_test)
    print('预测结果中各类数目:')
    print(pd.Series(val_pred).value_counts())
    print('\n F1 score为:')
    print(f1_score(train['label'].values,val_pred,average='macro'))

def SVM_Models(train):
    tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=5000)
    train_test=tfidf.fit_transform(train['text'].values.astype("U"))

    reg=svm.SVC(C=1.0,kernel='linear',degree=3,gamma='auto',decision_function_shape='ovr')
    reg.fit(train_test,train['label'].values)

    val_pred=reg.predict(train_test)
    print('预测结果中各类数目')
    print(pd.Series(val_pred).value_counts())
    print('\n F1 score为:')
    print(f1_score(train['label'].values,val_pred,average='macro'))

if __name__ == '__main__':
    train=read_csv(file='F:\data.csv')
    count_vectors_ridgeclassifier(train)
    tf_idf_ridgeclassifier(train)
    Parameter_regularization(train)
    Max_Features(train)
    Ngram_Range(train)
    LogisticRegression(train)
    SGDClassifier(train)
    SVM_Models(train)

