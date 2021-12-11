# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 14:53
# @Author  : ShaJiu
# @FileName: trainDevTest.py
# @Software: PyCharm
import codecs

"""
功能:平行语料按照训练、开发、测试集进行分配
    输入：双语平行语料
    输出：双语测试集、开发集、训练集
"""
file_read = [".//处理好的数据\\train.seq.en",
             ".//处理好的数据\\train.seq.zh"]

""""
读取文件
"""
with codecs.open(file_read[0], "r", encoding="utf-8") as f: tmp_src = [tmp.rstrip() for tmp in f]
with codecs.open(file_read[1], "r", encoding="utf-8") as f: tmp_tgt = [tmp.rstrip() for tmp in f]

""""
写dev文件
"""
file_dev = [file_read[0] + "_src_dev", file_read[1] + "_tgt_dev"]
file_dev_src = codecs.open(file_dev[0], "w", encoding="utf-8")
file_dev_tgt = codecs.open(file_dev[1], "w", encoding="utf-8")
"""
写test文件
"""
file_test = [file_read[0] + "_src_test", file_read[1] + "_tgt_test"]
file_test_src = codecs.open(file_test[0], "w", encoding="utf-8")
file_test_tgt = codecs.open(file_test[1], "w", encoding="utf-8")

"""
写train文件
"""
file_train = [file_read[0] + "_src_train", file_read[1] + "_tgt_train"]
file_train_src = codecs.open(file_train[0], "w", encoding="utf-8")
file_train_tgt = codecs.open(file_train[1], "w", encoding="utf-8")

count = 1
l1 = []
l2 = []
l3 = []
for tmp1, tmp2 in zip(tmp_src, tmp_tgt):
    if tmp1 and tmp2 and len(tmp_src) == len(tmp_tgt):
        tmp1 = tmp1.lstrip().rstrip()
        tmp2 = tmp2.lstrip().rstrip()
        ration = int(len(tmp_tgt) * 0.0005)  # 开发集和测试集在全数据中的占比例

        if count % ration == 0:
            l1.append(count)
            print(count, ration)
            print(count, "dev_src=", tmp1)
            file_dev_src.write(tmp1 + "\n")
            print(count, "dev_tgt=", tmp2)
            file_dev_tgt.write(tmp2 + "\n")
        elif count % (ration + 1) == 0:
            l2.append(count)
            print(count, ration + 1)
            print(count, "test_src=", tmp1)
            file_test_src.write(tmp1 + "\n")
            print(count, "test_tgt=", tmp2)
            file_test_tgt.write(tmp2 + "\n")
        else:
            l3.append(count)
            print(len(tmp_tgt))
            print(count, "train_src=", tmp1)
            file_train_src.write(tmp1 + "\n")
            print(count, "train_tgt=", tmp2)
            file_train_tgt.write(tmp2 + "\n")

        count += 1
print("开发集个数:", len(l1), "测试集个数:", len(l2), "训练集个数:", len(l3))
print("损失句子个数:", len(tmp_tgt) - len(l1) - len(l2) - len(l3))
