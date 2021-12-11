# -*- coding: utf-8 -*-
# @Time    :2018/12/14  22:05
# @Author  :shajiu
# @Site    :test01
# @File    :Python_workspace
# @Software:PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy

"""
功能:对平行语料进行打乱-有益更好的训练翻译模型
     输入平行语料
     输出打乱后的平行语料
"""
name = ["F:\测试数据\\train.bpe.tb",
        "F:\测试数据\\train.bpe.tb"]
# 平行语料文件路径

suffix = "shuf"  # 打乱文件的后缀
seed = 1234  # 打乱个数


def main():
    suffix = "." + "shuf"  # 打乱后的文件名称后缀
    stream = [open(item, "r", encoding="utf-8") for item in name]  # 读取两个文件并存入到stream中
    data = [fd.readlines() for fd in stream]  # 读取行并存入到data中去
    minlen = min([len(lines) for lines in data])  # 取出最短的句子长度

    if seed:  # 随机生成的数字（1234）args.seed
        numpy.random.seed(seed)  # 随机生成数字为0-1234  args.seed

    indices = numpy.arange(minlen)  # 随机生成以上所取出句子长度的最小值
    numpy.random.shuffle(indices)  # 打乱平行语料的顺序

    newstream = [open(item + suffix, "w", encoding="utf-8") for item in name]  # 写文件内容

    for idx in indices.tolist():  # 将数组或者矩阵转换成列表
        lines = [item[idx] for item in data]

        for line, fd in zip(lines, newstream):
            fd.write(line)  # 写入内容

    for fdr, fdw in zip(stream, newstream):
        fdr.close()
        fdw.close()


if __name__ == "__main__":
    main()
