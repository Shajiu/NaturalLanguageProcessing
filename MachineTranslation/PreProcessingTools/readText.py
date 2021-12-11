# -*- coding: utf-8 -*-
# @Time    : 2020/5/2 16:34
# @Author  : Shajiu
# @FileName: readText.py
# @Software: PyCharm
# @Github  ：https://github.com/Shajiu
import codecs
File1=codecs.open('E:\\1.txt','w',encoding='utf-8')
File2=codecs.open('E:\\2.txt','w',encoding='utf-8')
class Solution:
    def read(self,file):
        File=codecs.open(file,'r',encoding='utf-8')
        i=0
        for v in File:
            v=v.strip()
            if len(v)>0:
                if i%2==0:
                    File1.write(v+'\n')
                    print("藏文:",v)
                else:
                    File2.write(v + '\n')
                    print("中文:",v)
                i+=1

F='E:\\test.txt'
if __name__ == '__main__':
    result=Solution()
    print(result.read(F))