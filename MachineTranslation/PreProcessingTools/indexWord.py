#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 23:23
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: indexWord.py
# @Software: PyCharm
# @DATE:     2019/9/18

""""
功能：将编号写入的句子转换为词语
"""
import codecs

RAW_DATA = ".//train.count.zh"  # 原始的训练集数据文本
VOCAB = ".//vocab.zh"  # 生成的词汇表文件
OUTPUT_DATA = ".//train.word.zh"  # 将单词替换为单词编号后的输出文件

fin = codecs.open(RAW_DATA, "r", "utf-8")  # 读取单词文件
fout = codecs.open(OUTPUT_DATA, "w", "utf-8")  # 写入的文件

# 读取词汇表，并建立词汇到单词标号的映射
with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]  # 词汇表写入到元组中

word_to_id = {v: k for (v, k) in zip(range(len(vocab)), vocab)}  # 转换成字典的形式----键值对   字符---对应的行数


# 如果出现了被删除的低频词，则替换为"<unk>"
def get_id(words):
    sentence = []
    for word in words:
        word = int(word)  # 将字符转换为整数
        if word in word_to_id:
            sentence.append(word_to_id[word])
        else:
            sentence.append(word_to_id[0])
    sentence = sentence[:-1]  # 去除最后一个"<eos>"
    return sentence


if __name__ == '__main__':

    for line in fin:
        words = line.strip().split()  # 按照每一行的分词为单位写入到对应的列表中
        sentence = get_id(words)
        out_line = ' '.join([str(w) for w in sentence])
        fout.write(out_line + "\n")

    fin.close()
    fout.close()
