#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/1/13  9:56
# @Author  :shajiu
# @Site    :fenCi
# @File    :Python_workspace
# @Software:PyCharm
import jieba
import codecs
import collections
"""
功能：文本分词   读取文本   分词   写入文本
      输入:待分词的文本
      输出:已分词的文本
环境：python3、jieba、collections、codecs
"""

seg_list=jieba.cut("我来到北京清华大学",cut_all=True)
print("Full Mode:".join(seg_list))#全模式

seg_list=jieba.cut("我来到北京清华大学",cut_all=False)
print("Default Mode:","/".join(seg_list))#精确模式

seg_list=jieba.cut("他来到了网易杭研大厦")
print(",".join(seg_list))#默认是精确模式

seg_list=jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")#搜索引擎
print(".".join(seg_list))


RAW_DARA="./train.zh"    #待分词的数据文件
WRD_DATA="./train.seq.zh" #保存的数据路路径
counter=collections.Counter()
r='[’!"#$%&\'()*。、：‘丨’“”《》+,-.，/:;<=>?@[\\]^_`{|}~]+'   #这里可以加入不同的标点符号替换
write_file=open(WRD_DATA,"w",encoding="utf-8")
with codecs.open(RAW_DARA,"r","utf-8") as f:
    i=0
    for line in f:
        line=line.replace(" ","").replace("\n","")
        seg_list = jieba.cut(line,cut_all=False)
        string_tmp=" ".join(seg_list).strip()# 精确模式
        string_tmp=string_tmp.lstrip().rstrip()
        write_file.write(string_tmp+"\n")
        i+=1
    print("总共",i,"句子")
