# -*-coding:utf-8-*-
# Author: alphadl
# parallelCorpusClean.py 2018/8/7 17:13
import datetime
import codecs
import collections

"""
功能：   输入分词后的文本     读取文件必须放在项目的根目录下
         过滤特征有： 
         1.双语句子长度比率
         2.重复句子
"""

s_path = ".//train.bpe.tb.shuf"  # 源端语言
t_path = ".//train.bpe.zh.shuf"  # 目标语言
s_type = "ti"  # 源端类型
t_type = "zh"  # 目标类型

ratio = [0.0, 1000]  # 比例


class sent_clean(object):
    """
    处理单个句子
    """

    def __init__(self,
                 text="",
                 lang_set=[],
                 ):
        self.text = text  # 文本内容
        self.lang_set = lang_set  #

    # 计算文本的字符串的长度
    def count_length(self, text_temp, lang_type):  # 第三个调用函数
        '''
        :param text_temp: 需要统计长度的文本
        :param lang_type: 语种
        :return: int(长度)
        '''
        # 判断字符串是否为字符   语种是否为规定的语种
        assert type(text_temp) is str, "只能输入字符格式."  # 语种术语str
        assert lang_type in self.lang_set, "语种lang_type限定在(%s)." % (','.join(self.lang_set))  # 长度

        if lang_type in self.lang_set:
            return len(self.symbol_clean(text_temp, lang_type))  # 中日韩按照 字   #传给symbol_clean计算这个函数输出的字符串的长度
        elif lang_type is self.lang_set:
            return len(self.symbol_clean(text_temp, lang_type).split())  # 英文按照 词

    # 正则化文本字符串
    def symbol_clean(self, text_temp, lang_type):  # 第二个函数----接受每一行的文本、类型
        '''
        :param text_temp: 文本（含有特殊符号的）
        :param lang_type: 语种
        :return: str(没有特殊符号的文本)
        '''
        text_temp = str(text_temp)  # 字符转化
        # 判断为字符串
        assert type(text_temp) is str, "只能输入字符格式.当前句子为%s，格式为%s." \
                                       % (str(text_temp), str(type(text_temp)))
        # 判断语言类型
        assert lang_type in self.lang_set, "语种lang_type限定在(%s)." % (','.join(self.lang_set))

        import re
        # 利用正则化取出字符串中的特殊字符、并且字符类型属于规定的字符种类
        return re.sub(
            "[a-zA-Z\s+\.\!\/_,$%^*(+\"\'+——！，。？、~@#￥%……&*（）’! #$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^`{|}~]+", "",
            text_temp) if lang_type in self.lang_set else text_temp.replace("/[\W\s_]/g", "").strip().lower()


# 不匹配数字字母下划线     匹配任意空白字符


class corpus_clean(sent_clean):
    """
    处理语料
    """

    def __init__(self, src_path="",
                 tgt_path="",
                 src_type="",
                 tgt_type="",
                 filter_ratio=[0, 1],
                 src_save_path="",
                 tgt_save_path="",
                 src_discard_path="",
                 tgt_discard_path="",
                 ):
        super().__init__(lang_set=self._lang_set)
        self.filter_ratio = filter_ratio  # 比例 英文：中文 为1.3~1.8之间
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.src_type = src_type
        self.tgt_type = tgt_type
        self.src_save_path = src_save_path
        self.tgt_save_path = tgt_save_path
        self.src_discard_path = src_discard_path
        self.tgt_discard_path = tgt_discard_path

    _lang_set = ['zh', 'ja', 'kr', 'en', 'ti']

    corpus = {'raw_src': [], 'raw_tgt': [],
              'src': [], 'tgt': [],
              'src_save': [], 'tgt_save': [],
              'src_discard': [], 'tgt_discard': []}

    # 第一步读取文本
    def load_data(self, opt='src', lang_type='zh'):  # 默认为:源端、中文
        '''
        :param opt: 要加载的是src/tgt文件
        :param lang_type: 要加载的语种
        :return:
        '''

        # 为源端
        if opt == 'src':
            path = self.src_path

        # 为目端
        elif opt == 'tgt':
            path = self.tgt_path

        # 开始读取文件
        with codecs.open(path, 'rb+', 'utf-8') as fr:
            raw_lines = fr.readlines()
            print("正在读取%s。。。" % path)
            lines = [(self.symbol_clean(str(line), lang_type),

                      self.count_length(self.symbol_clean(str(line), lang_type), lang_type))  # （内层正则化取出）（外部计算字符串的长度）

                     for line in raw_lines]  # 每一行的文本读取到line中

            print("处理完毕。。。")
            self.corpus['raw_' + opt] = raw_lines  # 原先的文本赋值
            self.corpus[opt] = lines  # 正则化后的赋值

    # 去重的一种set()方式   本程序为使用---因为不能保证顺序
    def repeat_clean(self, list1, list2):
        unique = set()
        src_repeat_clean, tgt_repeat_clean = [], []
        assert len(list1) == len(list2) and len(list1) < self.window, "超过预设窗口%d" % self.window
        repeat_len = len(list1)

        # in

        for s, t in zip(list1, list2):
            unique.add((str(s), str(t)))
        repeat_clean_len = len(unique)

        # out

        for s, t in unique:
            src_repeat_clean.append(s)
            tgt_repeat_clean.append(t)
        return src_repeat_clean, tgt_repeat_clean, repeat_len - repeat_clean_len

    def filter(self):
        src_size = len(self.corpus['src'])  # 计算源长度
        tgt_size = len(self.corpus['tgt'])  # 计算目长度

        assert src_size == tgt_size and src_size != 0, "请将源端和目标端同时加载后再过滤语料。。。"

        print("-------------------------过滤参数如下---------------------------")
        print("源端文件路径：%s,目标端文件路径为%s" % (self.src_path, self.tgt_path))
        print("源端语料规模为：%d，目标端预料规模为：%d" % (src_size, tgt_size))
        print("%s到%s的过滤比例范围在%s" % (self.src_type, self.tgt_type, str(self.filter_ratio)))
        # print("去重窗口大小为%d"%self.window)
        print("--------------------------开始过滤------------------------------")

        cnt = 0  # 行数
        dis_cnt = 0  # 删除文件行数
        save_cnt = 0  # 正规文件行数
        for src_sent, tgt_sent in zip(self.corpus['src'], self.corpus['tgt']):

            # print(">>>正在处理第%d/%d句，句子为：\n\t%s<==>%s，句长分别为：%d，%d" % (
            #     cnt + 1, src_size, self.corpus['raw_src'][cnt], self.corpus['raw_tgt'][cnt], src_sent[1], tgt_sent[1]))
            if cnt % 1 == 0:
                print(">>>正在处理第%d/%d句，句子为：\n\t%s:%s \t%s:%s \t句长分别为：%d，%d\n" % (
                    cnt + 1, src_size,  # 行数、总句子长度
                    self.src_type, self.corpus['raw_src'][cnt],  # 种类、内容
                    self.tgt_type, self.corpus['raw_tgt'][cnt],  # 种类、内容
                    src_sent[1], tgt_sent[1]))  # 长度、

            if tgt_sent[1] == 0:  # 清除特殊符号后长度为零直接discard
                self.corpus['src_discard'].append(self.corpus['raw_src'][cnt])
                self.corpus['tgt_discard'].append(self.corpus['raw_tgt'][cnt])
                dis_cnt += 1  # 计算空的行

            elif self.filter_ratio[0] < float(src_sent[1]) / tgt_sent[1] < self.filter_ratio[1]:  # 源端和目标端的之间的比值区间
                self.corpus['src_save'].append(self.corpus['raw_src'][cnt])
                self.corpus['tgt_save'].append(self.corpus['raw_tgt'][cnt])
                save_cnt += 1  # 正规比例的存储
            else:
                self.corpus['src_discard'].append(self.corpus['raw_src'][cnt])
                self.corpus['tgt_discard'].append(self.corpus['raw_tgt'][cnt])
                dis_cnt += 1  # 删除的文本
            cnt += 1
        print("+++处理完成，保留了%d句，删除了%d句子，滤后比例为:%f+++" % (save_cnt, dis_cnt, float(save_cnt) / cnt))
        assert cnt == dis_cnt + save_cnt

        # 保存的文件的路
        self.src_save_path = self.src_save_path + '_%d_ratio%s' % (save_cnt, str(self.filter_ratio))
        self.tgt_save_path = self.tgt_save_path + '_%d_ratio%s' % (save_cnt, str(self.filter_ratio))
        self.src_discard_path = self.src_discard_path + '_%d_ratio%s' % (dis_cnt, str(self.filter_ratio))
        self.tgt_discard_path = self.tgt_discard_path + '_%d_ratio%s' % (dis_cnt, str(self.filter_ratio))

        print("源端和目标端有效句对写入到: %s 和 %s" % (self.src_save_path, self.tgt_save_path))
        print("源端和目标端丢弃句对写入到: %s 和 %s" % (self.src_discard_path, self.tgt_discard_path))

        # 写入文件
        with codecs.open(self.src_save_path, 'wb+', 'utf-8') as fw_src_save:
            with codecs.open(self.tgt_save_path, 'wb+', 'utf-8') as fw_tgt_save:
                with codecs.open(self.src_discard_path, 'wb+', 'utf-8') as fw_src_dis:
                    with codecs.open(self.tgt_discard_path, 'wb+', 'utf-8') as fw_tgt_dis:
                        fw_src_save.writelines(self.corpus['src_save'])
                        fw_tgt_save.writelines(self.corpus['tgt_save'])
                        fw_src_dis.writelines(self.corpus['src_discard'])
                        fw_tgt_dis.writelines(self.corpus['tgt_discard'])

        print("+++写入完毕！！+++")


cc = corpus_clean(src_path=s_path, tgt_path=t_path,  # 赋值源端文本、赋值目标端文本
                  src_type=s_type, tgt_type=t_type,  # 类型
                  filter_ratio=ratio,  # 比例
                  src_save_path=s_path + '_save',  # 保存
                  tgt_save_path=t_path + '_save',
                  src_discard_path=s_path + '_discard',
                  tgt_discard_path=t_path + '_discard')

load_time = datetime.datetime.now()  # 记开始的时间

print("<start>加载源端数据...")
cc.load_data(opt='src', lang_type=s_type)  # 调函数传参数  ------1

print("<start>加载目标端数据...")
cc.load_data(opt='tgt', lang_type=t_type)

finish_load_time = datetime.datetime.now()  # 结束的时间

print("数据加载完成，用时", float((finish_load_time - load_time).microseconds) / 1000000, "秒")

print("<start>开始过滤...")
filter_time = datetime.datetime.now()

cc.filter()  # 调用函数

finish_filter_time = datetime.datetime.now()
print("长度过滤完成，用时", float((finish_filter_time - filter_time).microseconds) / 1000000, "秒")

print("<start>开始去重...")
repeat_time = datetime.datetime.now()

# 去重后的文件
src_clean_list = []
tgt_clean_list = []
with codecs.open(cc.src_save_path, 'rb+', 'utf-8') as src_r:  # 读取保存的文件
    with codecs.open(cc.tgt_save_path, 'rb+', 'utf-8') as tgt_r:  # 读取保存的文件
        # read src
        src_raw_lines = src_r.readlines()
        src_lines = [(cc.symbol_clean(str(line), cc.src_type),
                      cc.count_length(cc.symbol_clean(str(line), cc.src_type), cc.src_type))
                     for line in src_raw_lines]

        # read tgt
        tgt_raw_lines = tgt_r.readlines()
        tgt_lines = [(cc.symbol_clean(str(line), cc.tgt_type),
                      cc.count_length(cc.symbol_clean(str(line), cc.tgt_type), cc.tgt_type))
                     for line in tgt_raw_lines]

        assert len(src_lines) == len(tgt_lines)
        print("去重前行数为：", len(src_lines))

        clean_pair = []  # 干净的一堆
        raw_pair = []  # 生成的一堆
        # in
        for index in range(len(src_lines)):
            clean_pair.append(str(src_lines[index][0]) + "<==>" + str(tgt_lines[index][0]))
            raw_pair.append(str(src_raw_lines[index]) + "<==>" + str(tgt_raw_lines[index]))

        # 不能用set去重，否则保留不了raw顺序，用dic
        pair_dic = collections.OrderedDict()
        for pair_index in range(len(clean_pair)):
            if clean_pair[pair_index] not in pair_dic.keys():
                pair_dic[str(clean_pair[pair_index])] = [1, str(raw_pair[pair_index])]

            else:
                pair_dic[(clean_pair[pair_index])][0] += 1

        # 抽出raw_pair,用np.arr操作容易memory error，换成循环操作
        # cleaned_pair = list(np.array(list(pair_dic.values()))[:, 1])
        cleaned_pair = [p[1] for p in pair_dic.values()]
        cleaned_len = len(pair_dic)

        assert len(cleaned_pair) == cleaned_len, (len(cleaned_pair), cleaned_len)

        with codecs.open(cc.src_save_path + '_' + str(cleaned_len) + '_clean_repeat', 'wb+', 'utf-8') as src_w:
            with codecs.open(cc.tgt_save_path + '_' + str(cleaned_len) + '_clean_repeat', 'wb+', 'utf-8') as tgt_w:
                print("去重后行数为：", cleaned_len)
                # out
                for p in cleaned_pair:
                    s, t = str(p).split("<==>")[0], str(p).split("<==>")[1]
                    src_clean_list.append(s)
                    tgt_clean_list.append(t)

                assert len(src_clean_list) == len(tgt_clean_list)

                src_w.writelines(src_clean_list)
                print("去重后的源端已经写入完成，路径在%s" % (cc.src_save_path + '_%d' % len(src_clean_list) + '_clean_repeat'))

                tgt_w.writelines(tgt_clean_list)
                print("去重后的目标端已经写入完成，路径在%s" % (cc.tgt_save_path + '_%d' % len(tgt_clean_list) + '_clean_repeat'))
                finish_repeat_time = datetime.datetime.now()
                print("去重完成，用时", float((finish_repeat_time - repeat_time).microseconds) / 1000000, "秒")
