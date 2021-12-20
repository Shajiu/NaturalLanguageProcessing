#### 基于BERT舆情分析的Demo实例
##### 1、概述
   Google发布了《Pre-training of Deep Bidirectional Transformers for Language Understanding》，一举刷新多项NLP领域记录后。BERT为“Bidirectional Encoder Representations from Transformers”的首字母缩写，整体式一个自编码语言模型(Autoencoder LM)，而且其内部设置了两个任务来预测训练模型。
 * 第一个任务是采用MaskLM的方式来训练语言模型，通俗地讲就是在输入一句话的时候，随机地选一些预测的词，然后用一个特殊的符号[MASK]来代替他们，之后让模型根据所给定的标签去学习这些地方该填写的词；
 * 第二个任务在双向语言模型的基础上额外增加了一个句子级别的连续性预测任务，即预测输入BERT的两段文本是否为连续的文本，引入这个任务可以更好地让模型学到连续的文本片段之间的关系。

   按照[原论文](https://arxiv.org/abs/1810.04805)中的描述，最后的实验结果表明BERT模型的有效性，并在11项NLP任务上取得SOTA结果。BERT相较于原来的RNN,LSTM可以做到并发执行，同时提取词在句子中的关系特征，并且能在多个不同层次提取关系特征，进而更全面反映句子语义。相较于word2vec, 其又能根据句子上下文获取词义，从而避免歧义出现。同时缺点也是显而易见的，模型参数太多，而且模型太大，少数数据训练时，容易过拟合。

  BERT模型可谓红遍NLP领域，更多人都想通过BERT对自己的数据进行预处理。由于当前对BERT分析的文章太多，这里也不在赘述。本文接下来将会简单介绍几个使用BERT模型运行的Demo。
##### 2、实操
- 运行平台：Linux、1080Ti、Python=3.6.3$(Anaconda)$、Tensorflow=1.15.0。
- 下载源码：[BERT](https://github.com/google-research/bert)。或者直接使用clone方式获取源码。
```git
git clone https://github.com/google-research/bert.git
 ```
- 下载中文预训练摩模型[BERT-Base, Chinese:](https://link.csdn.net/?target=https%3A%2F%2Fstorage.googleapis.com%2Fbert_models%2F2018_11_03%2Fchinese_L-12_H-768_A-12.zip)，至于其他的预训练模型，请到[官网](https://github.com/google-research/bert)进行下载即可。其中模型文件夹下包含三个文件，依次为```bert_model.ckpt、 vocab.txt和bert_config.json```，其中```bert_model.ckpt```为预训练模型结果文件；```vocab.txt```为词汇表文件；```bert_config.json```为模型的超参数文件。

![BERT模型说明](https://s2.loli.net/2021/12/18/RtqxjKweh9yAHa4.jpg)
- 下载训练数据，本次举一个例子即可，本文按照官网说明进行下载MRPC语料。直接运行```download_glue_data.py```即可下载GLUE data。具体执行命令如下：
``` Python
python3 download_glue_data.py --data_dir glue_data --tasks MRPC
 ```
   然后这样的方式下载80%都为失败，并且下载的部分也比较慢，可以采用文末提供的链接即可下载全部文件。最终此文件夹下包含如下文件```dev.tsv、dev_ids.tsv、msr_paraphrase_test.txt、msr_paraphrase_train.txt、test.tsv、train.tsv、xx.tsv```。
#####  3、Run Demo
基于MRPC语料的句子对分类分类任务
**训练**：在bert源码文件里执```run_classifier.py```，基于预训练模型进行Fine-tune，直接运行run_train.sh即可:```./run_train.sh```，其中run_train.sh中的具体命令如下：
``` Sh
 python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=glue_data/MRPC \
  --vocab_file=models/cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=models/cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=models/cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=emotion_output \
  --do_lower_case=True \
  --do_lower_case=False
 ```
其中，–do_eval为true的意思，即会生成模型验证结果文件如下图。如果false，既没有验证结果生成。最终的模型保存在```output_dir```中，验证结果为:
```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
  ```
模型具体存储文件如下：
![存储的模型文件](https://s2.loli.net/2021/12/18/CqmBwOv8En17SNb.jpg)
**验证:** 指定Fine-tune之后模型所在地，执行```run_test.sh```，其中```run_test.sh```中的具体内容如下:
``` Python
python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=glue_data/MRPC \
  --vocab_file=models/cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=models/cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=emotion_output \
  --max_seq_length=128 \
  --output_dir=./tmp/mrpc_output/
  ```
##### 4、自定义任务/加载数据
本次实例以情感分析为例，进行实验，本次主要修改两处。
**构建数据:** 以情感分析为例，以句子级别为粒度的构建数据集，每条序列的分类-正/中/负，分别用“0、1、2”表示。则构建的数据集如下：
![一行一条，序列与标签之间用"\t"分割](https://s2.loli.net/2021/12/18/dYKEZamge1AnTvb.jpg)
**代码修改:** 第一处是在在```run_classifier.py```中添加读取自己的数据集即可。原先脚本中读取数据的类分别有``` XnliProcessor、MnliProcessor、MrpcProcessor、ColaProcessor```。首先加入一个类，用于读取自定义数据集，此处所加的类为```EmotionProcessor```，具体内容如下：
``` python
class EmotionProcessor(DataProcessor):
  """针对情感分析进行训练的测试数据的读取"""
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
  def get_labels(self):
    """See base class."""
    return ["0","1","2"]
  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
 ```
 解释，其中以上代码中，我们的标签为正/中/负，所以:
 ```
 def get_labels(self):
    """See base class."""
    return ["0","1","2"]
  ```
本此数据中每行只有一条数据和标签(没有别的列)，其中数据列为第0列，标签列为第1列，所以对应的设置如下：
```python
text_a = tokenization.convert_to_unicode(line[0])
label = tokenization.convert_to_unicode(line[1])
text_b=None
 ```
 第二处修改是在``` run_classifier.py的processors```中加入```"emotion":EmotionProcessor```最终为如下内容:
```
processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "emotion":EmotionProcessor
  }
  ```
其中```emotion```为执行脚本时传入的参数，类似于任务名称；```EmotionProcessor```为以上添加的读取自定义数据的类名。接下来即可执行。执行分别为微调和验证，可以修改以上```run_train.sh和run_test.sh```中的输入/输出路径即可。
##### 5、获取实例源码
以上实例的所有源码以及数据可以在[这里](https://github.com/Shajiu/NaturalLanguageProcessing/tree/master/SentimentAnalysis/BERT)获取，其中获取中文预训练模型，请[点击](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)这里。
##### 6、总结
本文对BERT做了一些简单的Demo展示, 首先，介绍了BERT中的两个重要任务，分别为Masked LM和Next Sentence Prediction；其次，介绍了如何下载源码/预训练模型并介绍基本的配置环境需求；然后，通过官网说明下载MRPC语料并进行了微调和预测；最后，以情感分析的数据实例进行了详细的自定义数据微调和预测的使用说明。希望通过简要介绍能为一些类似本人的小白起到学习的作用。
