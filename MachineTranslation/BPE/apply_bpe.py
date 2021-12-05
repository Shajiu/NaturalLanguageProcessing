#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich
# flake8: noqa
# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 23:23
# @USER:     ShaJiu
# @Author  : shajiu
# @FileName: index_word.py
# @Software: PyCharm
# @DATE:     2019/9/18

from __future__ import unicode_literals, division
import codecs
import argparse
import re
from io import open
argparse.open = open

"""
功能: 对文本进行BPE处理化
     输入： 对应的词频统计
     训练出来的模型(BPE)
     输出:BPE处理完的文本
"""

file_codes="./models/bpe32k"
merges=-1
separator="@@"
file_vocabulary="G://vocab.en"
vocabulary_threshold=50
glossaries=None
class BPE(object):
    def __init__(self, codes, separator='@@', vocab=None, glossaries=None):


        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(
                r'(\.0+)*$', '', firstline.split()[-1]).split(".")])
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict(
            [(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict(
            [(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.cache = {}

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        output = []
        for word in sentence.split():
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return ' '.join(output)

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                             for out_segments in isolate_glossary(segment, gloss)]
        return word_segments



def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries=None):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    if orig in glossaries:
        cache[orig] = (orig,)
        return (orig,)

    if version == (0, 1):
        word = tuple(orig) + ('</w>',)
    elif version == (0, 2):  # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + (orig[-1] + '</w>',)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)

    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item


def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.split()
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary


def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.

    Returns a list of subwords. In which all 'glossary' glossaries are isolated 

    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    if word == glossary or glossary not in word:
        return [word]
    else:
        splits = word.split(glossary)
        segments = [segment.strip() for split in splits[:-1]
                    for segment in [split, glossary] if segment != '']
        return segments + [splits[-1].strip()] if splits[-1] != '' else segments

"""句子级别的处理"""
def read_sentence(tmp):
    codes = codecs.open(file_codes, encoding='utf-8')
    vocabulary = codecs.open(file_vocabulary, encoding='utf-8')

    vocabulary = read_vocabulary(
        vocabulary, vocabulary_threshold)

    bpe = BPE(codes, separator, vocabulary, glossaries)

    tmp=bpe.segment(tmp).strip()
    return tmp

"""句子级别的处理"""
def read_doc(file_input,file_output):
    codes = codecs.open(file_codes, encoding='utf-8')
    input = codecs.open(file_input, encoding='utf-8')
    output = codecs.open(file_output, 'w', encoding='utf-8')
    vocabulary = codecs.open(file_vocabulary, encoding='utf-8')

    vocabulary = read_vocabulary(
        vocabulary, vocabulary_threshold)

    bpe = BPE(codes, separator, vocabulary, glossaries)
    for line in input:
        output.write(bpe.segment(line).strip())
        output.write('\n')

    return "处理完毕"

if __name__ == '__main__':

    """句子级别的处理"""
    input = ["ཨུ་ རུ་ སུའི་ གསར་བརྗེ འི་ གཞི་རྩའི", "ངས་ ཁྱོད་ ལ་ རོགས་བྱེད་ ཐུབ་ བམ",
             "ཐེངས་ བཅུ་དྲུག་པ འི་ སྙན་ཞུ འི་ ནང་ གང་ལེགས་ མཚོན་ དགོས", "ངས་ རིན་པ་ བྱིན་ ན་ ཆོག་ གམ"]

    count=0
    for tmp in input:
        tmp=read_sentence(tmp)
        count+=1
        print("处理",count,"--->",tmp)

    """文本级别处理"""
    file=["G://input_test.txt","G://output_test.txt"]
    read_doc(file[0],file[1])