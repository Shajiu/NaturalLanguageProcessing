import argparse
import json
import copy
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader
import utils
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

    def train(self, epoch, data_loader): 
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            loss_list.append(loss.cpu().item())           

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def sample_difficulty(self,epoch,loss_train_list):
        # 计算样本难度: 返回的值越小越好
        loss=dict()
        if epoch==0:
            for index in loss_train_list[epoch]:
                loss[index]=loss_train_list[epoch][index]
        else:
            for index1,index2 in zip(loss_train_list[epoch-1],loss_train_list[epoch]):
                loss1=loss_train_list[epoch-1][index1]
                loss2=loss_train_list[epoch][index2]
                loss[index1]=(loss1-loss2)/loss1           
        # print("测试段:\t","第",epoch,"次迭代的第",i,"个样本的前后loss为:",loss)
        return loss
        

    def model_competence(self,f1_t):
        # 计算模型能力
        # C=min(1,F1_t*(1-c0)/(F1_best*p)+c0)
        return  min(1,f1_t*(1-config.c0)/(config.f1_best*config.p)+config.c0)
    
    def sort_train(self,result_loss):
        # 根据每个训练样本的难度对训练数据进行排序
        return sorted(result_loss.items(),key=lambda x:x[1],reverse=False)
    

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        data_loss=dict()
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
                data_loss[i]=loss.cpu().item()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return e_f1,data_loss

    def predict(self, epoch, data_loader, data):
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
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def  load_dataset(config,sort_list,c):
    datasets, ori_data = data_loader.load_data_bert(config,sort_list,c)
    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                batch_size=config.batch_size,
                collate_fn=data_loader.collate_fn,
                shuffle=False,
                num_workers=0,
                drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )
    return datasets,train_loader, dev_loader,test_loader,ori_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/example.json')
    parser.add_argument('--save_path', type=str, default='./data/example/Test/models/model.pt')
    parser.add_argument('--predict_path', type=str, default='./data/example/Test/result/output.json')
    parser.add_argument('--device', type=int, default=5)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--f1_best', type=int,default=0.75,help="Per-train a vanilla NER model and record the best f1 value on the development set as curriculum length f1_best")
    parser.add_argument('--c0', type=int,default=0.2,help="Intialize model competence is estimated")
    parser.add_argument('--p', type=int,default=0.9,help="(0,1] is a coefficient to control the curriculum speed")

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets,train_loader, dev_loader,test_loader,ori_data=load_dataset(config,sort_list={},c=0)
    inference_data=copy.deepcopy(train_loader)
    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info("Building Model")
    model = Model(config)
    model = model.cuda()
    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
  
    train_loss=dict()
    for epoch in range(config.epochs):
        logger.info("Epoch: {}".format(epoch))
        trainer.train(epoch, train_loader)
        f1,_= trainer.eval(epoch, dev_loader)
        logger.info("Dev f1 is:\n{}".format(f1)) 
        c=trainer.model_competence(f1)
        logger.info("epoch={}\t model competence:{}".format(epoch,c))

        _,train_loss[epoch]=trainer.eval(epoch, inference_data)        
        result_loss=trainer.sample_difficulty(epoch,train_loss)
        result_loss=trainer.sort_train(result_loss)

        test_f1,_= trainer.eval(epoch, test_loader, is_test=True)      
        
        if f1 > best_f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)
        _,train_loader,_,_,_=load_dataset(config,sort_list=result_loss,c=c)
        
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1])

