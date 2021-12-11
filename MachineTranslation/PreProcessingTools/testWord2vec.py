# -*- coding: utf-8 -*-
# @Time    : 2019/8/11 10:32
# @Author  : ShaJiu
# @FileName: testWord2vec.py
# @Software: PyCharm
"""
功能:使用Word2Vec使用的方式---
     模型训练、模型测试
"""

# 第一种训练方式
# encoding=utf-8

"""
第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5,
第三个参数是神经网络的隐藏层单元数，默认为100
"""
from gensim.models import word2vec

sentences = word2vec.Text8Corpus(u"F:\测试数据\\train.seq.clean.ti")
model = word2vec.Word2Vec(sentences, min_count=5, size=50)
y2 = model.similarity(u"好", u"还行")
print(y2)

for i in model.most_similar(u"滋润"):
    print(i[0], i[1])

# 模型使用
"""
根据词向量求相似
"""
model.similarity("first", "is")  # 两个词的相似性距离
model.most_similar(positive=["first", "second"], negative=["sentence"], topn=1)  # 类比的防护
model.doesnt_match("input is lunch he sentence cat".split())  # 找出不匹配的词语

# 词向量查询
model_tmp = model["first"]
print(model_tmp)

# 模型保存与加载
from gensim.models import KeyedVectors

# save
model.save("fname")  # 只有这样才能继续训练
model.wv.save_word2vec_format("outfile" + ".model.bin", binary=True)  # C binary format 磁盘空间比上一方法减半
model.wv.save_word2vec_format("outfile" + ".model.txt", binary=False)  # C text format 磁盘空间大，与方法一样

# load
# 最省内存的加载方法
model = word2vec.load('model path')
word_vectors = model.wv
del model
word_vectors.init_sims(replace=True)
