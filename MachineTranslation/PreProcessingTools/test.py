#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020-1-8 17:11
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: wangyi-1.py
# @Software: PyCharm
# @DATE:     2020-1-8

"""构建格式"""


def translation_inputs(id, sentences):
    inpus = []
    for v in sentences:
        d = {}
        d["src"] = v
        inpus.append(d)
    tmp = inpus[0]
    tmp["id"] = id
    inpus[0] = tmp
    return inpus


tmp = ['And now for something completely different .',
       'I love you .']

if __name__ == '__main__':
    tmps = translation_inputs(1, tmp)
    print(tmps)

tmp = "And now for something completely different. I love you. "
from nltk.tokenize import sent_tokenize

sent = sent_tokenize(tmp)
print(sent)
