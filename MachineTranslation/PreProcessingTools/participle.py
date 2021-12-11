#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/3 22:49
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: participle.py
# @Software: PyCharm
# @DATE:     2019/11/3 
"""
功能：英文分词工具
      输入:英文待分词的文本
      输出:英文已分词后的文本
环境：python3、spacy、codecs
"""
import spacy
import codecs

File=["./train.en","./train.token.de"]
out_file=codecs.open(File[1],"w",encoding="utf-8")

"""具体读取"""
def read_file():
    with codecs.open(File[0], "r", "utf-8") as f:
        for line in f:
            nlp = spacy.load('en_core_web_sm')
            token = nlp(line)
            tmp = " ".join(t.text for t in token).lstrip().rstrip()
            out_file.write(tmp + "\n")
            print("分词后的文本:",tmp)

if __name__ == '__main__':
    read_file()
    print("分词结束啦")

