#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/1 19:54
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: attention.py
# @Software: PyCharm
# @DATE:     2019/12/1 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import codecs

"""针对源语言和目标语言画出Attention注意机制图像
   输入：注意机制权重矩阵
        源语言、目标语言
   输出：图像
"""

def read_attention(file):
    src = []
    tgt = []
    attentions = []
    count = 0
    file = codecs.open(file, "r", encoding="utf-8")
    for tmp in file:
        tmp = ' '.join(tmp.split())
        tmp = tmp.split()
        if count == 0:
            src = tmp
        else:
            tgt.append(tmp[0])
            attention = []
            for tm in tmp[1:]:
                tm = tm.replace("*", "")
                tm = float(tm)
                tm = round(tm, 2)
                attention.append(tm)
            attentions.append(attention)
        count += 1
    attentions_np = np.array(attentions)
    attentions_np = attentions_np.reshape(len(tgt), len(src))  # 行、列
    return attentions_np, src, tgt


def seaborn_heatmap(attention, src, tgt):
    data = pd.DataFrame(attention, columns=src, index=tgt)
    ax = sns.heatmap(data.T, square=True, center=0)
    ax.set_xlabel('TargetSsentence', fontdict={'family': 'Times New Roman', 'size': 10})  # 源语言
    ax.set_ylabel('SourceSentence', fontdict={'family': 'Times New Roman', 'size': 10})  # 目标语言

    for item in ax.get_yticklabels():
        item.set_rotation(0)
        item.set_fontname('Times New Roman')

    for item in ax.get_xticklabels():
        item.set_rotation(10)
        item.set_fontname('Times New Roman')

    """具体设置"""
    plt.rcParams["figure.figsize"] = (2.0, 5.0)
    plt.title('Attention Mechanism Visualization(Fr-En)', y=1.05, fontdict={'family': 'Times New Roman', 'size': 10})
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.tick_params(labelsize=10)
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    file="./data/attention.txt"
    attentions, src, tgt = read_attention(file)
    seaborn_heatmap(attentions, src, tgt)
