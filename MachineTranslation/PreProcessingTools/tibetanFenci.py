#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/4/24;18:39
# @Author  :shajiu
# @Site    :PyCharm
# @File    :pycharm_workspace
# @Software:PyCharm
import codecs

"""

功能：为藏文分词的预处理和后期处理工作所使用
      读取数据并去除文本行末尾的空格字符串
      输入文本文件
      输出文本文件
"""

file = ["F:\北理实验室项目\网上资料\数据文件\藏汉平行语料集\其他平行语料汇总\处理过测\高质量对齐句子\\6-酥油灯法律\\6-train.txt[分词].txt"]
write_file = open("F:\北理实验室项目\网上资料\数据文件\藏汉平行语料集\其他平行语料汇总\处理过测\高质量对齐句子\\6-酥油灯法律\\6-train.seq.tb",
                  "w", encoding="utf-8")

"""分词前的处理"""


def read_1():
    with codecs.open(file[0], "r", encoding="utf-8") as rf:
        cum = 1
        for tmp in rf.reader:
            aa = tmp.strip()
            print(cum, aa)
            aa = aa.replace(" ", "")
            aa = aa.lstrip().rstrip()
            write_file.write(aa + "<沙九>" + "\n")
            cum += 1


"""分词后"""


def read_2():
    with codecs.open(file[0], "r", encoding="utf-8") as rf:
        cum = 1
        for tmp in rf.reader:
            for aa in tmp.split("</沙九/>"):
                aa = aa.replace("/", " ")
                aa = aa.replace(" >", ">")
                aa = aa.lstrip().rstrip()
                write_file.write(aa + "\n")
                print("行数", cum)
                cum += 1


if __name__ == '__main__':
    # read_1()  #分词前的处理
    read_2()  # 分词后的处理
