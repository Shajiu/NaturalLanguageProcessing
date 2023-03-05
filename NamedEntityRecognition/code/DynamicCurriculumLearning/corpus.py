# -*- coding = utf-8 -*-
# @time:2022/8/9 下午9:20
# Author:sha jiu
# @File:corpus.py
# @Software:PyCharm
import codecs
import json

def read_file(file):
    with codecs.open(file,'r',encoding="utf-8") as file:
        result_list = list()
        for lin in file.readlines():
            data=dict()
            lin=eval(lin.rstrip())
            data["sentence"] = [v for v in lin["text"]]
            ner=list()
            for v in lin["entities"]:
                if v["start_offset"]<=v["end_offset"]:
                    tmp=dict()
                    tmp["index"] = [v for v in range(v["start_offset"], v["end_offset"]+1)]
                    tmp["type"]=v["label"]
                    ner.append(tmp)
            data["ner"]=ner
            result_list.append(data)
        #print(result_list)
        write_file(result_list)

def write_file(result_list):
    json.dump(result_list, out_file,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    input_file="E:\PycharmProjects\paddleuie\CMeEE\\英文版本\\"
    out_file=codecs.open("/googol/nlp/shajiu/W2NER/data/example/test.json",'w',encoding="utf-8")
    read_file(input_file)