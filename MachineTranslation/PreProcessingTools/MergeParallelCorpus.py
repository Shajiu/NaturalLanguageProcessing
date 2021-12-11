# -*- coding: utf-8 -*-
# @Time    : 2020/5/2 15:47
# @Author  : Shajiu
# @FileName: MergeParallelCorpus.py
# @Software: PyCharm
# @Github  ：https://github.com/Shajiu
import codecs


class Solution:
    def read(self, f1, f2, f3):
        File1 = codecs.open(f1, 'r', encoding='utf-8')
        File2 = codecs.open(f2, 'r', encoding='utf-8')
        File3 = codecs.open(f3, 'w', encoding='utf-8')
        for val1, val2 in zip(File1, File2):
            val1 = val1.strip().replace(' ', '')
            val2 = val2.strip().replace(' ', '')
            tmp = 'Source:' + val1 + 'Target:' + val2
            File3.write(tmp + '\n')
        return 0


file = ['E:\\中-1.txt',
        'E:\\藏-1.txt',
        'E:\\合并.txt']
if __name__ == '__main__':
    result = Solution()
    print(result.read(file[0], file[1], file[2]))
