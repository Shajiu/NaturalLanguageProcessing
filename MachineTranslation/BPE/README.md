切分字词的程序、将把BPE原始码很好的简化和浓缩，可便于入门者理解和使用。
- 1、[apply_bpe.py:](https://github.com/Shajiu/NaturalLanguageProcessing/blob/master/MachineTranslation/BPE/apply_bpe.py)对文本进行BPE处理化,输入： 对应的词频统计、训练出来的模型(BPE)、输出:BPE处理完的文本。
- 2、[learn_joint_bpe_and_vocab.py:](https://github.com/Shajiu/NaturalLanguageProcessing/blob/master/MachineTranslation/BPE/learn_joint_bpe_and_vocab.py)词频统计、输入数据：切分好的训练数据、输出数据: 词频统计、输入格式: 
``` shell 
--input corpus.tc.tb corpus.tc.zh -s 32000 -o bpe32k --write-vocabulary vocab.tb vocab.zh
 ```
- 3、输入源语言、目标语言、32000 -o bpe32k、源语言词频统计表、目标词频统计表。


