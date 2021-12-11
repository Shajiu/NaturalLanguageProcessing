#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 23:04
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: wordIndex.py
# @Software: PyCharm
# @DATE:     2019/9/18 

""""
功能：对文本编号： 对文本中的每个单词进行编号，所编的号码是对应的词在词汇表中的位置的行数
"""
import codecs

RAW_DATA=".//test.tb"#原始的训练集数据文本
VOCAB=".//vocab.src"#生成的词汇表文件
OUTPUT_DATA=".//test.src"#将单词替换为单词编号后的输出文件

#读取词汇表，并建立词汇到单词标号的映射
with codecs.open(VOCAB,"r","utf-8") as f_vocab:
    vocab=[w.strip() for w in f_vocab.readlines()]   #转换
word_to_id={k:v for (k,v) in zip(vocab,range(len(vocab)))}   #转换成字典的形式----键值对   字符---对应的行数
#如果出现了被删除的低频词，则替换为"<unk>"
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


fin=codecs.open(RAW_DATA,"r","utf-8")   #读取单词文件
fout=codecs.open(OUTPUT_DATA,"w","utf-8")  #写入的文件

for line in fin:
    words=line.strip().split()+["<eos>"] #读取单词并添加<eos>结束符    按照每一行的分词为单位写入到对应的列表中并在末尾加入<eos>
    #将每个单词替换为词表中的编号
    out_line=' '.join([str(get_id(w)) for w in words])+"\n"
    fout.write(out_line)
fin.close()
fout.close()