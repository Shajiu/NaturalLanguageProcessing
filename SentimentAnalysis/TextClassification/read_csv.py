# -*- coding: utf-8 -*-
# @Time    : 2020/10/2 23:53
# @Author  : Shajiu
# @FileName: read_csv.py
# @Software: PyCharm
# @Github  ：https://github.com/Shajiu
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def read_csv(file):
    return pd.read_csv(file,sep=',',nrows=15000)

def Bags_of_Words():
    corpus=['ཤེས་རྟོགས་ བྱུང་ བ ར་ གཞིགས་ ན། ད་ལོ་ མངའ་རིས་ ས་ཁུལ་ གྱིས་ རིགས་ འདྲ་མིན',
            'ཞིང་འབྲོག་པར་ རྩལ་ནུས་ སྦྱོང་བརྡར་ སྤྲོད་ པ འི་ ཁྲོད་ མངའ་རིས་ ས་ཁུལ་ གྱིས་ ཞིང་ལས་ དང་ འབྲེལ་ བ འི་ ཚན་པས་ ཞིང་འབྲོག་པ',
            'ད་ལོ་ མངའ་རིས་ ས་ཁུལ་ གྱིས་ ད་དུང་ སྔ་རྗེས་ སུ་ ལས་ཞུགས་ རོགས་སྐྱོར་ ཟླ་བ་ ཞེས་ པ་ དང',
            'ཤེས་རྟོགས་ བྱུང་ བ ར་ གཞིགས་ན། འདི་ ཟླ འི་ ཟླ་མཇུག་ ཏུ་ མགོ་རྡོག་ ཆུ་ཤུགས་ གློག་ཁང་ གི་ འཕྲུལ་ཚོ་ ཐོག་མ']
    vectorizer=CountVectorizer()
    vectorizer.fit_transform(corpus).toarray()
    print(vectorizer.fit_transform(corpus).toarray())

if __name__ == '__main__':
    data=read_csv('./corpus/data.csv')
    Bags_of_Words()

