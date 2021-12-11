#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2019/5/21;11:28
# @Author  :shajiu
# @Site    :PyCharm
# @File    :pycharm_workspace
# @Software:PyCharm
import codecs

""""
功能:处理测试数据:将正规的数据转化为标准的测试数据--为CWMT2018测试工具准备数据
    输入:数据测试集文件
    输出:标准的XMl格式数据   源文:srcset   参考:refset   译文:tstset
"""
input = ".//CWMT_2019\结果\CCMT2019_009_TC\\translation.zh"
output = ".//CWMT_2019\结果\CCMT2019_009_TC\\tc-2019-hit-primary-b.xml"

Prefix = """
<?xml version="1.0" encoding="UTF-8"?>
<srcset setid="ti_zh_govdoc_trans" srclang="ti" trglang="zh">
<DOC docid="govdoc"  site="1" refid="1">
<p>
"""
suffix = """
</p>
</DOC>
</srcset>
"""


def read_txt():
    with codecs.open(input, "r", encoding="utf-8") as f:
        i = 0
        list_tmp = []
        for tmp in f:
            i += 1
            tmp = tmp.replace("\n", "").strip()
            tmp = tmp.replace(" ", "")
            tmp = ''.join(['<seg id="' + str(i) + '">' + tmp + '</seg>'])
            list_tmp.append(tmp)
        return list_tmp


def write_txt(tmp):
    with codecs.open(output, "w", encoding="utf-8") as f:
        f.write(Prefix)  # 开始部分
        for tm in tmp:
            f.write(tm + "\n")
        f.write(suffix)  # 结尾部分


if __name__ == '__main__':
    tmp = read_txt()
    write_txt(tmp)
