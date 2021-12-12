# -*- coding: utf-8 -*-
# @Time    : 2020/10/2 21:30
# @Author  : Shajiu
# @FileName: data.py
# @Software: PyCharm
# @Github  ：https://github.com/Shajiu
import os
import codecs
import pandas as pd

def read_data(path):
    corpus = []
    label=[]
    for file in os.listdir(path):
        if not os.path.isdir(file): #判断是否为文件夹，不是文件夹才打开
            if 'txt' in file:
                for v in codecs.open(path + "/" + file, 'r', encoding='utf-8'):
                    v=v.strip()
                    corpus.append(v)
                    label.append(file.split('.')[0])
    dataframe = pd.DataFrame({'text': corpus, 'label':label})
    dataframe.to_csv('./corpus/data.csv',index=False,sep=',',encoding='utf-8')

if __name__ == '__main__':
    result=read_data(path='./corpus/')