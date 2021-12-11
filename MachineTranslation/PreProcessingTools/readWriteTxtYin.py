#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/3/15  10:56
# @Author  :shajiu
# @Site    :read_write_txt
# @File    :pycharm_workspace
# @Software:PyCharm
import codecs
import re

"""
     功能:平行语料按照字符或者音节形式分割
          输入：未切分的平行语料（一种）   
          输出：按音节或者字符为粒度切分的文本
          注意：在主函对应的换藏文和中文
"""

r_flie = "C:\\Users\shajiu\Desktop\毕业设计\德吉拉毛\\新建文本文档.txt"
w_file = "C:\\Users\shajiu\Desktop\毕业设计\德吉拉毛\\test.txt"

write_file = codecs.open(w_file, "w", "utf-8")

"""中文"""


def read_zh(read_file):
    with codecs.open(read_file, "r", "utf-8") as ff:
        i = 1
        for line in ff:
            line = line.replace(" ", "")
            tmp = ' '.join([a for a in line]).rstrip().lstrip()
            tmp = re.sub(" +", "", tmp)
            print(i, tmp)
            tmp = tmp.rstrip().lstrip()
            write_file.write(tmp + "\n")
            i += 1


"""藏文"""


def read_tb(read_file):
    with codecs.open(read_file, "r", "utf-8") as ff:
        i = 0
        for line in ff:
            tmp = '་ '.join([a for a in line.split("་")]).rstrip().lstrip()
            tmp = tmp.replace("  ", " ")
            tmp = re.sub(" +", "", tmp)
            if len(tmp.split()) == 2:
                tmp = tmp.split()[1]
                write_file.write(tmp + "\n")
                i += 1


# 主函数
if __name__ == '__main__':
    read_zh(r_flie)
    read_tb(r_flie)
