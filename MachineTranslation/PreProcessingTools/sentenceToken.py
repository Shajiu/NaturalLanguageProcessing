#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/17 15:43
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: sentenceToken.py
# @Software: PyCharm
# @DATE:     2019/11/17
"""
功能:  英文分词、英文分句、中文分词、识别语种
       输入：待分句/分词/识别语种的文本
       输出：已分句/分词/识别的文本

环境：spacy、nltk、langid、jieba
"""
import spacy
from nltk.tokenize import sent_tokenize
import jieba
import langid

text1="There are never any lights there, except the lights that we " \
    "bring. We 're designed by nature to play from birth to old age ."

text2="由宗教引起的矛盾和冲突打着宗。教旗号进行的侵略和战争多得很"

text3="སདབ་གདབ་གངལཕཙེརཏིོཕཆེཏིོཕཆེར་གཁླྑཤཟཐཇཀགལཞའསཏབགལཞེདརིབཏསགདབལཞའསདིགབོཕསགཕ"

"""英文分句子"""
def phrasing_sentence(txt):
    nlp = spacy.load('en_core_web_sm')
    token = nlp(txt)
    tmp = " ".join(t.text for t in token).lstrip().rstrip()
    return tmp

"""英文分词"""
def token_sentence(txt):

    """英文分词"""
    ans_list = sent_tokenize(txt)
    print("英文分词",ans_list)

    """"中文分词"""
    seg_list = jieba.cut(txt, cut_all=False)
    tmp=" ".join(seg_list)
    print("中文分词:" + tmp)  # 精确模式
    return tmp

"""识别语种"""
def langid_sentence(text):
    array = langid.classify(text)
    print("语种:",array[0])
    print("语种为:",array[0])
    return array[0]

if __name__ == '__main__':
    sen1=phrasing_sentence(text1)
    print("分句后的文本内容:",sen1)
    sen2=token_sentence(text2)
    print("分词后的文本内容:",sen2)
    sen3=langid_sentence(text3)
    print("识别语种后的文本:",sen3)



