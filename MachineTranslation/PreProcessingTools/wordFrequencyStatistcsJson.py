#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/6/6;14:11
# @Author  :shajiu
# @Site    :PyCharm
# @File    :pycharm_workspace
# @Software:PyCharm
import json
import codecs
""""
功能:  写json数据
       输入:词汇表  file[0] counts 
       输出:json格式的词汇表file[1]      
"""
file=["G:\\vocab.de",
      "G:\\vocab-de.json"]
counts=50000 #词汇表的大小
write_file=codecs.open(file[1],"w",encoding="utf-8")
def read():
    dict={}
    with codecs.open(file[0],"r",encoding="utf-8") as f:
        id=0
        for tmp in f.reader:
            tmp=tmp.strip()
            dict[tmp]=id  #添加内容
            id+=1
            if id>counts:
                break
    string_json=json.dumps(dict, ensure_ascii=False,sort_keys=True,indent=2,separators=(",",": "))
    print(string_json)
    write_file.write(string_json)

if __name__ == '__main__':
    read()
