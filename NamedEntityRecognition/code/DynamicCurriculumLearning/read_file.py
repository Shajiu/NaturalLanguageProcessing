import json 
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
import os
from torch.utils.data import DataLoader
from model import Model
import argparse
import torch
import torch.nn as nn
import transformers
class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.save_path = config["save_path"]
        self.predict_path = config["predict_path"]

        self.dist_emb_size = config["dist_emb_size"]
        self.type_emb_size = config["type_emb_size"]
        self.lstm_hid_size = config["lstm_hid_size"]
        self.conv_hid_size = config["conv_hid_size"]
        self.bert_hid_size = config["bert_hid_size"]
        self.biaffine_size = config["biaffine_size"]
        self.ffnn_hid_size = config["ffnn_hid_size"]

        self.dilation = config["dilation"]

        self.emb_dropout = config["emb_dropout"]
        self.conv_dropout = config["conv_dropout"]
        self.out_dropout = config["out_dropout"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.warm_factor = config["warm_factor"]

        self.use_bert_last_4_layers = config["use_bert_last_4_layers"]

        self.seed = config["seed"]

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/example.json')
parser.add_argument('--save_path', type=str, default='./model.pt')
parser.add_argument('--bert_learning_rate', type=float)
parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")
args = parser.parse_args()
args = parser.parse_args()
config =Config(args)

class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length):
        self.bert_inputs = bert_inputs
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item]

    def __len__(self):
        return len(self.bert_inputs)




os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]       
        pieces = [piece for pieces in tokens for piece in pieces]
        
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        length = len(instance['sentence'])
        if length>512:
            print("序列长度超过512:",length)
            continue
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)
        
        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k
             
        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
    return bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        print("测试段:",label)
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def load_data_bert():
    with open('/googol/nlp/shajiu/W2NER/data/example/w2ner_ext_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("/googol/nlp/plm_models/ernie-health-zh", cache_dir="./cache/")

    vocab = Vocabulary()
    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return test_dataset, test_data


test_dataset, test_data=load_data_bert()

updates_total =22870

print(test_dataset)
print(test_data)
print(len(test_dataset))





from model import Model

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)


    def load(self, path):
        print("模型路径:",path)
        self.model.load_state_dict(torch.load(path))

    def predict(self, epoch, data_loader, data,lab="F"):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                print("-----------------------------------")
                sentence_batch = data[i:i+config.batch_size]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                if lab=="F":
                    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
                else:
                    bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)

                decode_entities = utils.decode_test(outputs.cpu().numpy(), length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        for x in ent[0]:
                            pass
                            #print("text:",[x for x in ent[0]])
                            #print("type:",config.vocab.id_to_label(ent[1]))

                    #     instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                    #                                "type": config.vocab.id_to_label(ent[1])})
                    # result.append(instance)

                # total_ent_r += ent_r
                # total_ent_p += ent_p
                # total_ent_c += ent_c

                #grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

               # label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        #label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)


model = Model(config)


model = model.cuda()

trainer = Trainer(model)

best_f1 = 0
best_test_f1 = 0
trainer.load(config.save_path)
trainer.predict("Final", test_dataset, test_data,lab="F")




