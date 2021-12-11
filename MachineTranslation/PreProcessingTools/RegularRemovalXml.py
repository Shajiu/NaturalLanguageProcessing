#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/4/21;21:50
# @Author  :shajiu
# @Site    :PyCharm
# @File    :pycharm_workspace
# @Software:PyCharm
import codecs
import re

"""
功能:
     正则表达---通过匹配提取文本内容-----从标准的测试数据集中可以删除xml格式提取正文内容
     去除文本开头空格、去除文本末尾空格
     从xml文本中提取内容写入到txt中整理
"""
read_file = "F:\北理实验室项目\网上资料\数据文件\WMT2014平行数据\处理好的-en-fr\WMT-14-EN-FR\\test\\newstest2014-fren-ref.fr.sgm"
write_file = open("F:\北理实验室项目\网上资料\数据文件\WMT2014平行数据\处理好的-en-fr\WMT-14-EN-FR\\test\\newstest2014.fr.sgm", "w",
                  encoding="utf-8")
with codecs.open(read_file, "r", encoding="utf-8") as rf:
    i = 0
    for tmp in rf.readlines():
        n = re.findall(r"<[^>]*>", tmp)
        if "</seg>" in n:
            res = re.sub("<[^>]*>", "", tmp)
            tmp = res.split("\n")[0]
            print(i, n, tmp)
            tmp = tmp.lstrip()  # 去除字符串开头的空格
            tmp = tmp.rstrip()  # 去除字符串末尾的空格
            write_file.write(tmp + "\n")
            i += 1
