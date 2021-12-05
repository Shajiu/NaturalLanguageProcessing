#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/1 21:39
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: testBleu.py
# @Software: PyCharm
# @DATE:     2019/12/1 
import matplotlib.pyplot as plt

# 月份
test_name = ['En-De(dev)', 'En-De(test)', 'De-En(dev)', 'De-En(test)', 'En-Vi(dev)', 'En-Vi(test)', 'Vi-En(dev)', 'Vi-En(test)',
      'En-Fr(dev)', 'En-Fr(test)', 'Fr-En(dev)', 'Fr-En(test)']

# 体重

bleu1=[22.92,22.81,29.97,29.84,23.88,22.58,22.71,22.98,36.53,36.42,35.95,34.11]
bleu2=[23.33,23.14,30.43,30.24,24.06,22.72,23.07,23.46,37.45,37.32,36.81,34.92]
bleu3=[23.72,23.41,31.29,31.05,24.78,23.52,23.58,24.12,38.18,37.95,37.46,35.55]

# 设置画布大小
plt.figure(figsize=(16, 4))
font = {'family': 'Times New Roman',
         'size': 16,
         }
#
A, =plt.plot(test_name, bleu1, 'b:*',ms=10,label = 'TF',)
B, =plt.plot(test_name, bleu2, 'r*:', ms=10,label = 'STC',)
C, =plt.plot(test_name, bleu3, 'g:*', ms=10,label = 'UC-LE',)

#   Approach For Low Resource Domain Specific Terminology Neural Machine Translation

#   Training Neural Machine Translation To Apply Terminology Constraints
# 设置图例并且设置图例的字体及大小
plt.legend(handles=[A, B, C], prop=font)
plt.xticks(rotation=25)
plt.xlabel("DifferentTestSets",fontdict={'family' : 'Times New Roman', 'size':16})                     # X轴标签
plt.ylabel("BLEU",fontdict={'family' : 'Times New Roman', 'size'   : 16})                                 # Y轴标签
plt.title("BleuAtDifferentMethod",fontdict={'family' : 'Times New Roman','size': 16})    # 标题
plt.savefig("./result/easyplot.png")                                                                             # 存储文件
plt.legend()
plt.show()