#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 23:21
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: wordStatistics.py
# @Software: PyCharm
# @DATE:     2019/9/18

import codecs
import collections
from operator import itemgetter

"""
功能：对分词后的文本进行词频统计 选出前5000个词汇写入一个文本中
"""

word_max = 100000  # 词频统计的个数
RAW_DARA = ".//corpus.tc.de"  # 训练集的数据文件
VOCAB_OUTPUT = ".//vocab.de"  # 输出的词汇表文件

counter = collections.Counter()
i = 0
with codecs.open(RAW_DARA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():  # 按空格分词进行分行
            counter[word] += 1
# 按词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
# 稍后我们需要在文本换行处加入句子结束符<eos>,这里预先将其加入到词汇表中
sorted_words = ["<eos>"] + sorted_words
sorted_words = ["<unk>", "<sos>,", "<eos>"] + sorted_words
print(sorted_words)
print(len(sorted_words))

if len(sorted_words) > word_max:
    sorted_words = sorted_words  # [:100000000]
with codecs.open(VOCAB_OUTPUT, "w", "utf-8") as file_output:
    for word in sorted_words:
        print(word)
        file_output.write(word + "\n")
