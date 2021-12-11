#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 22:09
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: divideSentence.py
# @Software: PyCharm
# @DATE:     2019/10/22
"""
功能：英文文本分句
      输入:待分句的英文文本
      输出:已分句好的文本
环境：
    java1.8、python3;

"""

import os
def divide_sentence(str):
    sentence_list=[]
    input_file = open("./input.en","w",encoding="utf-8")
    input_file.write(str)
    input_file.close()
    f = os.popen(r"java -jar divide_sentence.jar ./input.en","r",)

    for lines in f:
        sentence_list.append(lines.rstrip())
        print("测试段---:",lines.rstrip())
    return sentence_list

if __name__ == '__main__':
    txt = "red sky， bleu sky. yes no is. "
    ans = divide_sentence(txt)
    print(ans)
