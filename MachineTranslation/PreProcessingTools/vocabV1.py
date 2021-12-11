# -*- coding: utf-8 -*-
# @Time    :2018/12/14  22:05
# @Author  :shajiu
# @Site    :test01
# @File    :Python_workspace
# @Software:PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections

"""
功能:为RNN-NMT-xmunmt创建词汇表
     输入为：两种训练语料合在一起的数据
     输出为：词汇表
"""
file_input=".//corpus.bpe32k.all"  #训练语料
file_output=".//corpus-tb-zh.txt" #词汇表
string_tmp="</s>,UNK"#特殊字符串
def count_words(filename):
    counter = collections.Counter()
    with open(filename, "r",encoding="utf-8") as fd:
        for line in fd:
            words = line.strip().split()
            counter.update(words)

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))
    print("测试段",count_pairs)
    print("测试段1:",words)
    print("测试段2:",counts)
    return words, counts

def control_symbols(string):
    if not string:
        return []
    else:
        return string.strip().split(",")


def save_vocab(name, vocab):
    if name.split(".")[-1] != "txt":
        name = name + ".txt"

    pairs = sorted(vocab.items(), key=lambda x: (x[1], x[0]))
    words, ids = list(zip(*pairs))

    with open(name, "w",encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")

def main(): #args
    vocab = {}
    #limit = args.limit
    limit=0
    count = 0
    words, counts = count_words(file_input) #args.corpus
    ctrl_symbols = control_symbols(string_tmp)#args.control

    for sym in ctrl_symbols:
        vocab[sym] = len(vocab)

    for word, freq in zip(words, counts):
        if limit and len(vocab) >= limit:
            break

        #如果字符在字典中、则跳过添加
        if word in vocab:
            print("Warning: found duplicate token %s, ignored" % word)
            continue

        vocab[word] = len(vocab)
        count += freq
    save_vocab(file_output, vocab)#args.output

    print("Total words: %d" % sum(counts))
    print("Unique words: %d" % len(words))
    print("Vocabulary coverage: %4.2f%%" % (100.0 * count / sum(counts)))


if __name__ == "__main__":
    #main(parse_args())
    main()  #直接调用函数
