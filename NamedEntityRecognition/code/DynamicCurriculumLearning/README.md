# DynamicCurriculumLearning

此项目为基于动态课程学习实现的NER或者span-tagging方法，具体参考文献为《Dynamic Curriculum Learning for LOw-Resource Neural Machine Trainslation》。
- 核心逻辑： 人类在学习过程中，仅利用少量的数据就可以学习到很好的水平，受到人类学习策略的启发，我们使用课程学习方法来解决低资源问题。通过一种从易到难的学习方式。循序渐进的学习方式。
- 每个阶段训练数据的选取满足如下两个准则： ①样本难度；②模型能力。具体通过损失下降速度来衡量样本难度，通过基于验证集上的F1值动态计算模型能力。

### 1、环境需求
##### 1.1 必须具备如下需求
```
- python (3.8.12)
- cuda (11.4)
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```
##### 1.2 或者直接使用已经克好的docker镜像，镜像命名为:```harbor.unisound.ai/shajiu/cuda:w2enr```。
### 2、数据格式
##### 2.1 数据格式为json,具体实例在此框架下```data/example1```中，其中train.json、dev.json和test.json分别表示训练集、开发集和测试集。如下所示:
```
[{"sentence": ["Muscle", "Pain", "."], "ner": [{"index": [0, 1], "type": "ADR"}]}]
```
##### 2.2 当前模型仅仅支持序列长度小于等于512个字符串，且框架中具备去除过滤机制

### 3、训练过程
##### 3.1 训练时的参数说明
```
--config  配置文件，默认为```./config/example.json```
--save_path 模型存储路径，默认为：```./data/example/Test/models/model.pt```
--predict_path 预测之后的结果输出文件，默认为: ```./data/example/Test/result/output.json```
--device 指定GPU编号，默认为0
--f1_best 训练普通NER或者span-tagging后在开发集上的最佳f1值，小数点保留两位数(非百分制)
--c0 初始化模型能力，默认为0.2，最初模型从20%最简单的句子开始学习
--p  学习进度，默认为0.9，表示当模型性能(f1)达到0.9时，使用全部训练集进行学习
```
##### 3.2 具体训练命令
``` 
python train.py --config ./config/example.json
```
##### 3.3 训练阶段输出日志
```
shajiu@48f8e12bbc89:/googol/nlp/shajiu/DynamicCurriculumLearning$ python train.py
2023-03-01 07:12:14 - INFO: dict_items([('dataset', {'train': 'example/Test/train.json', 'valid': 'example/Test/valid.json', 'test': 'example/Test/test.json'}), ('save_path', './data/example/Test/models/model.pt'), ('predict_path', './data/example/Test/result/output.json'), ('dist_emb_size', 20), ('type_emb_size', 20), ('lstm_hid_size', 512), ('conv_hid_size', 128), ('bert_hid_size', 768), ('biaffine_size', 556), ('ffnn_hid_size', 384), ('dilation', [1, 2, 3, 4]), ('emb_dropout', 0.5), ('conv_dropout', 0.5), ('out_dropout', 0.33), ('epochs', 10), ('batch_size', 1), ('learning_rate', 0.001), ('weight_decay', 0), ('clip_grad_norm', 5.0), ('bert_name', '/googol/nlp/plm_models/ernie-health-zh'), ('bert_learning_rate', 5e-06), ('warm_factor', 0.1), ('use_bert_last_4_layers', False), ('seed', 123), ('config', './config/example.json'), ('device', 5), ('f1_best', 0.95), ('c0', 0.2), ('p', 0.9)])
2023-03-01 07:12:14 - INFO: Loading Data
2023-03-01 07:12:15 - INFO:
+-------------------------+-------+-----------+----------+
|           path          |  type | sentences | entities |
+-------------------------+-------+-----------+----------+
| example/Test/train.json | train |    2131   |   7309   |
| example/Test/valid.json | valid |    300    |   1023   |
|  example/Test/test.json |  test |    300    |   1019   |
+-------------------------+-------+-----------+----------+
2023-03-01 07:18:48 - INFO: Building Model
Some weights of BertModel were not initialized from the model checkpoint at /googol/nlp/plm_models/ernie-health-zh and are newly initialized: ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
2023-03-01 07:18:58 - INFO: Epoch: 0
2023-03-01 07:28:23 - INFO:
+---------+--------+--------+-----------+--------+
| Train 0 |  Loss  |   F1   | Precision | Recall |
+---------+--------+--------+-----------+--------+
|  Label  | 0.0386 | 0.3479 |   0.3422  | 0.3815 |
+---------+--------+--------+-----------+--------+
/opt/conda/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
2023-03-01 07:29:13 - INFO: EVAL Label F1 [0.99994933 0.55062473 0.        ]
2023-03-01 07:29:13 - INFO:
+--------+--------+-----------+--------+
| EVAL 0 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.5169 |   0.6192  | 0.4685 |
| Entity | 0.0000 |   0.0000  | 0.0000 |
+--------+--------+-----------+--------+
2023-03-01 07:29:14 - INFO: Dev f1 is:
0
2023-03-01 07:29:14 - INFO: epoch=0      model competence:0.2
/opt/conda/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set les. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
2023-03-01 07:35:52 - INFO: EVAL Label F1 [0.99994804 0.53024281 0.        ]
2023-03-01 07:35:52 - INFO:
+--------+--------+-----------+--------+
| EVAL 0 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.5101 |   0.6175  | 0.4616 |
| Entity | 0.0000 |   0.0000  | 0.0000 |
+--------+--------+-----------+--------+
2023-03-01 07:35:53 - INFO:
Before selection:426    Training data size:2131
2023-03-01 07:35:53 - INFO:
+-------------------------+-------+-----------+----------+
|           path          |  type | sentences | entities |
+-------------------------+-------+-----------+----------+
| example/Test/train.json | train |    426    |   793    |
| example/Test/valid.json | valid |    300    |   1023   |
|  example/Test/test.json |  test |    300    |   1019   |
+-------------------------+-------+-----------+----------+
/opt/conda/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set les. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
2023-03-01 07:39:41 - INFO: TEST Label F1 [0.99994285 0.49773399 0.        ]
2023-03-01 07:39:41 - INFO:
+--------+--------+-----------+--------+
| TEST 0 |   F1   | Precision | Recall |
+--------+--------+-----------+--------+
| Label  | 0.4992 |   0.6168  | 0.4506 |
| Entity | 0.0000 |   0.0000  | 0.0000 |
+--------+--------+-----------+--------+
```
### 4、推理说明
##### 4.1 推理默认加载test数据进行推理，输出为
```{"text": ["骨", "水", "泥"], "type": "植入物"}, {"text": ["骨", "水", "泥"], "type": "植入物"}, {"text": ["骨", "水", "泥"], "type": "植入物"}```
##### 4.2 具体推理命令为：```python test.py --config ./config/example.json

##### 4.3 若想修改另外文件进行推理或者修改输出文件格式，请在```test```文件中的```predict```函数中进行修改即可。





