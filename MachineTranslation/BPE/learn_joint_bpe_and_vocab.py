#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
This script learns BPE jointly on a concatenation of a list of texts (typically the source and target side of a parallel corpus,
applies the learned operation to each and (optionally) returns the resulting vocabulary of each text.
The vocabulary can be used in apply_bpe.py to avoid producing symbols that are rare or OOV in a training text.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""
"""
功能: 词频统计
      输入数据：切分好的训练数据
      输出数据: 词频统计
      输入格式: --input corpus.tc.tb corpus.tc.zh -s 32000 -o bpe32k --write-vocabulary vocab.tb vocab.zh
                        输入源语言、目标语言         32000 -o bpe32k                    源语言词频统计表   目标词频统计表
"""
import sys
import os
import inspect
import codecs
import argparse
import tempfile
import warnings
from collections import Counter
from learn_bpe import learn_bpe,get_vocabulary
from apply_bpe import BPE
from io import open
argparse.open = open

"""基本参数"""
file_input=["F:\北理实验室项目\网上资料\数据文件\CCMT_2019\dev2019\QHNU-test-tizh-CWMT2018\CWMT2018-TestSet-TC\\deve.seq.tb",
            "F:\北理实验室项目\网上资料\数据文件\CCMT_2019\dev2019\QHNU-test-tizh-CWMT2018\CWMT2018-TestSet-TC\\deve.seq.zh"]                #平行训练语料
file_output="./models/bpe32k"                                    #BPE模型
symbols=32000                                          #词汇表大小
separator='@@'                                         #符号表示
write_vocabulary=["G://vocab.en","G://vocab.de"]       #词频统计
min_frequency=2
total_symbols=False
verbose=False

def learn_joint_bpe_and_vocab():

    if write_vocabulary and len(file_input) != len(write_vocabulary):
        sys.stderr.write('Error: number of input files and vocabulary files must match\n')
        sys.exit(1)

    # read/write files as UTF-8
    input = [codecs.open(f, encoding='UTF-8') for f in file_input]
    vocab = [codecs.open(f, 'w', encoding='UTF-8') for f in write_vocabulary]

    # get combined vocabulary of all input texts
    full_vocab = Counter()
    for f in input:
        full_vocab +=get_vocabulary(f)         #传输文件名称
        f.seek(0)

    vocab_list = ['{0} {1}'.format(key, freq) for (key, freq) in full_vocab.items()]

    # learn BPE on combined vocabulary
    with codecs.open(file_output, 'w', encoding='UTF-8') as output:
        """调用函数"""
        learn_bpe(vocab_list, output, symbols, min_frequency, verbose, is_dict=True, total_symbols=total_symbols)

    with codecs.open(file_output, encoding='UTF-8') as codes:
        """调用函数"""
        bpe = BPE(codes, separator=separator)

    # apply BPE to each training corpus and get vocabulary
    for train_file, vocab_file in zip(input, vocab):

        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()

        tmpout = codecs.open(tmp.name, 'w', encoding='UTF-8')

        train_file.seek(0)
        for line in train_file:
            tmpout.write(bpe.segment(line).strip())
            tmpout.write('\n')

        tmpout.close()
        tmpin = codecs.open(tmp.name, encoding='UTF-8')

        vocab = get_vocabulary(tmpin)          #调用函数
        tmpin.close()
        os.remove(tmp.name)

        for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write("{0} {1}\n".format(key, freq))
        vocab_file.close()



if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.simplefilter('default')
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    assert(len(file_input) == len(write_vocabulary))
    learn_joint_bpe_and_vocab()
