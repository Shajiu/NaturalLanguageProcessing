# -*- coding: utf-8 -*-
# @Time    :2018/12/14  22:05
# @Author  :shajiu
# @Site    :test01
# @File    :Python_workspace
# @Software:PyCharm
import codecs
import collections
from operator import itemgetter
"""
功能：对分词后的文本进行词频统计
      文本word.txt中为关键词 词频
      文本test.vocab.txt中为选出前1000个词汇排序后
"""
RAW_DARA=".//en-de\原始数据\\train\\train.en"    #分词的数据文件
WORD_OUTPUT=".//en-de\原始数据\\vocab\\count.en.txt"  #输出的词频表文件
VOCAB_OUTPUT=".//en-de\原始数据\\vocab\\vocab.en.txt"  #输出的词汇表文件
counter=collections.Counter()
with codecs.open(RAW_DARA,"r","utf-8") as f:
    for line in f:
        for word in line.strip().split():     #按空格分词进行分行
            counter[word]+=1
    for k in counter.keys():
        pass
       # print(counter[k])
#        print(k)
with codecs.open(WORD_OUTPUT, "a", "utf-8") as ff:
    for k in counter:
         ff.write(k+",")
         ff.write(str(counter[k])+"\n")

#按词频顺序对单词进行排序
sorted_word_to_cnt=sorted(counter.items(),key=itemgetter(1),reverse=True)
sorted_words=[x[0] for x in sorted_word_to_cnt]
#稍后我们需要在文本换行处加入句子结束符<eos>,这里预先将其加入到词汇表中
#sorted_words=["<eos>"]+sorted_words
#sorted_words=["</s>,UNK"]+sorted_words #"<unk>","<s>",
if len(sorted_words) > 8000000:
    sorted_words=sorted_words[:8000000]
with codecs.open(VOCAB_OUTPUT,"a","utf-8") as file_output:
    for word in sorted_words:
        file_output.write(word+"\n")