#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 22:09
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: divideSentence.py
# @Software: PyCharm
# @DATE:     2019/10/22 

"""第一种"""
from nltk.tokenize import sent_tokenize
ans_list = sent_tokenize("xuda is a boy. he is happy.")
print(ans_list)



"""第二种"""
import spacy
nlp = spacy.load('en_core_web_sm')
st="xuda is a boy. he is happy."
document = nlp(st)
print(document)
