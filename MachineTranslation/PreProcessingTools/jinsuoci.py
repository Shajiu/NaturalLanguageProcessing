# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 17:19
# @Author  : ShaJiu
# @FileName: jinsuoci.py
# @Software: PyCharm
"""
paper：藏文紧缩格识别方法   功能:识别紧缩词功能
五元法；；；；；不够的前面加start、后面加end
"""
file = "F:\测试数据\\deve.seq.tb"


def read():
    list = []
    with open(file, "r", encoding="utf-8") as f:
        for vale in f.readlines():
            vale = vale.rstrip().lstrip()
            list.append(vale)
    return list


def sentence(vale):
    for v in vale:
        v = v.replace(" ", "")
        v = v.split("་")
        function1(v)


""""
算法1: 拟紧缩音节识别算法
"""
def function1(sentence):
    tmp1 = ["ག", "ང", "བ", "མ"]
    tmp2 = ["དག", "དང", "དབ", "དམ", "འག", "འབ", " མག"]
    sentence_tmp = []
    for Words in sentence:
        if len(Words) > 1 and "ས" or "ར" in Words:
            if len(Words) >= 3 and Words[-2:-1] in tmp1:
                if len(Words) == 3 and Words[:-1] in tmp2:
                    sentence_tmp.append(Words+"yT")
                    print(Words, "为拟紧缩词")
                else:
                    sentence_tmp.append(Words + "nT")
                    print(Words, "不是拟紧缩词")
            else:
                sentence_tmp.append(Words + "yT")
                print(Words, "为拟紧缩词")
        else:
            sentence_tmp.append(Words + "nT")
            print(Words, "非拟紧缩词")
    print("识别为:",sentence_tmp)


"""
算法2 紧缩格的规则识别算法
"""


def function2():
    sentence = "wwwww"  # 为拟紧缩格的五元处理对象
    tmp3 = ["གིས་", "ཀྱིས་", "གྱིས་", "འིས་", "ཡིས་"]
    tmp4 = ["ངས་", "འདིས་", "དེར་", "འདིར་"]
    tmp5 = ["ཨ་མྱེས།ཨ་གར།ཨ་ཀར།ཨ་གསར།ཨ་སྒོར།ཨ་འཐས།ཨ་ཟེར།ཨ་བཅས།ཨ་བར།"]
    DEI = ["单音节、双音节、三音节中拟紧缩格都在任何情况下为后加字的词1622"]
    """"
    ཤེས།ཆོས།གོས་བཅོས།འགུས་རིས།ལས་གཞི།ཟུར་མིག།སློང་གསོ་ཅུས།བདག་གིར་བཞེས།མར་ཁེ་སི།དམར་པོ་རི།
    """
    if sentence[2] in tmp3:
        print(sentence[2], "为非紧缩词")
    elif sentence[2] in tmp4:
        print(sentence[2], "为紧缩词")
        if sentence[1] == "ཨ":
            if sentence[1] in tmp5:
                print(sentence[2], "为非紧缩词")
            else:
                print(sentence[2], "为紧缩词")
        elif sentence[2] or sentence[1:2] or sentence[:2] or sentence[2:3] or sentence[2:] in DEI:
            print(sentence[2], "为紧缩词")
        else:
            print(sentence[2], "无法辨别")


""""
算法3   紧缩格的添加-还原算法
"""


def function3():
    i = 0
    sentence = ["wi-2", "wi-1", "wi", "wi+1", "wi+2"]  # w1为拟紧缩格的五元处理对象
    word1 = sentence[i]  # "拟紧缩格"
    word2 = sentence[i - 1] + sentence[i]  # "拟紧缩格"
    word3 = sentence[i - 2] + sentence[i - 1] + sentence[i]  # "拟紧缩格"
    DB2 = "存放最后一个音节为午后加字或者后加字为“འ的二音节字、三音节"
    """
    རྡོ་རྗེ།མཁན་པོ།ནམ་མཁའ།ཕ་མཐའ།རིན་པོ་ཆེ།ག་གེ་མོ།སའི་གོ་ལ།
    """
    if word1[-1] or word1[-2] in "下加字或者元音或者上加字":
        if word2 or word3 in DB2:
            print(sentence[i], "为紧缩词")
        else:
            print(sentence[i], "不确定啦")
    elif word2 + "འ" or word3 + "འ" in DB2:
        print(sentence[i], "为紧缩词")
    else:
        print(sentence[i], "无法判断啦")


if __name__ == '__main__':
    vale = read()
    sentence(vale)
