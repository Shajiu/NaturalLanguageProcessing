# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 16:54
# @Author  : ShaJiu
# @FileName: sentenceCleanShuffle.py
# @Software: PyCharm
import codecs

"""
功能:过滤文件
     输入文件:平行语料
              长句子、短句子、中间句子
"""
file = ["F:\北理实验室项目\网上资料\数据文件\藏汉平行语料集\其他平行语料汇总\处理过测\高质量对齐句子\最终版(1-6)\\train.seq.tb",
        "F:\北理实验室项目\网上资料\数据文件\藏汉平行语料集\其他平行语料汇总\处理过测\高质量对齐句子\最终版(1-6)\\train.seq.zh"]

"""
读取文件
"""
with codecs.open(file[0], "r", encoding="utf-8") as f: list_src = [tmp.rstrip() for tmp in f]
with codecs.open(file[1], "r", encoding="utf-8") as f: list_tgt = [tmp.rstrip() for tmp in f]

"""
写短句
"""
file_short_src = codecs.open(file[0] + "_short_src", "w", encoding="utf-8")
file_short_tgt = codecs.open(file[1] + "_short_tgt", "w", encoding="utf-8")

"""
写标准句
"""
file_standatd_src = codecs.open(file[0] + "_standatd_src", "w", encoding="utf-8")
file_standatd_tgt = codecs.open(file[1] + "_standatd_tgt", "w", encoding="utf-8")

"""
写长句
"""
file_long_src = codecs.open(file[0] + "_long_src", "w", encoding="utf-8")
file_long_tgt = codecs.open(file[1] + "_long_tgt", "w", encoding="utf-8")

count_short = 1
count_standard = 1
count_long = 1
count = 1
for tmp1, tmp2 in zip(list_src, list_tgt):
    tmp1 = tmp1.lstrip().rstrip()
    tmp2 = tmp2.lstrip().rstrip()

    if tmp1 and tmp2 and len(list_src) == len(list_tgt):
        print(count, tmp1, tmp2)
        count += 1
        if len(tmp1) <= 15 and len(tmp2) <= 15:
            print(count_short, tmp1, tmp2)
            file_short_src.write(tmp1 + "\n")
            file_short_tgt.write(tmp2 + "\n")
            count_short += 1

        elif len(tmp1) <= 200 and len(tmp2) <= 200:
            print(count_standard, tmp1, tmp2)
            file_standatd_src.write(tmp1 + "\n")
            file_standatd_tgt.write(tmp2 + "\n")
            count_standard += 1

        else:
            print(count_long, tmp1, tmp2)
            file_long_src.write(tmp1 + "\n")
            file_long_tgt.write(tmp2 + "\n")
            count_long += 1

print("sentences number", count, "sentences_short", count_short, "sentences_standatd", count_standard, "sentences_long",
      count_long, )
print("loss sentences", count - (count_standard + count_long + count_short))
