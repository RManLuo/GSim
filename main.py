#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 10:06
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : main.py
# @Software: PyCharm
# @Describe:
import os
from statistics import mean

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import torch
from utils.helper import setup_seed
import dgl
from models import ContextGNN
from evaluate import evaluate_task
from utils import load_data, EarlyStopping, load_latest_model, evaluate, recall_evaluator
from utils.preprocess import SimPairDataset
import argparse
from pytorch_metric_learning import losses
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from pytorch_metric_learning import distances
import scipy.sparse as sp
import time

parser = argparse.ArgumentParser(description='Which GPU to run?')
parser.add_argument('-n', default=0, type=int, help='GPU ID')
args = parser.parse_args()

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def model_evaluate(model, g, features, labels, mask, loss_func, p_nodes_id):
    model.eval()
    with torch.no_grad():
        all_emb = model(g, features)
    p_emb = all_emb[p_nodes_id]
    logits = model.get_predict(p_emb[mask])
    loss = loss_func(logits, labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits, labels[mask])

    return loss, accuracy, micro_f1, macro_f1


last_memory =0
def get_memory_total():
    global last_memory
    last_memory = torch.cuda.memory_allocated() / 1024 / 1024
    return last_memory

def get_memory_diff():
    last = last_memory
    total = get_memory_total()
    return total - last, total

def add_inter_node(hg, dataset='DBLP'):
    if dataset == 'DBLP':
        e_num = hg.number_of_edges('pa') + hg.number_of_edges('pc') + hg.number_of_edges('pt')
        ind = hg.incidence_matrix(typestr='out', etype='pa')._indices()
        row, col = ind[0], ind[1]

        ind_1 = hg.incidence_matrix(typestr='out', etype='ap')._indices()
        row_1, col_1 = ind_1[0], ind_1[1]
        values_1 = torch.Tensor([1] * len(col_1))
        ae_adj = sp.csr_matrix((values_1, (row_1, col_1)), shape=(hg.number_of_nodes('a'), e_num),
                               dtype=np.float64)

        ind = hg.incidence_matrix(typestr='out', etype='pc')._indices()
        row = torch.cat([row, ind[0]])
        col = torch.cat([col, ind[1] + hg.number_of_edges('pa')])
        ind_1 = hg.incidence_matrix(typestr='out', etype='cp')._indices()
        row_1, col_1 = ind_1[0], ind_1[1] + hg.number_of_edges('pa')
        values_1 = torch.Tensor([1] * len(col_1))
        ce_adj = sp.csr_matrix((values_1, (row_1, col_1)), shape=(hg.number_of_nodes('c'), e_num),
                               dtype=np.float64)

        ind = hg.incidence_matrix(typestr='out', etype='pt')._indices()
        row = torch.cat([row, ind[0]])
        col = torch.cat([col, ind[1] + hg.number_of_edges('pc') + hg.number_of_edges('pa')])
        values = torch.Tensor([1] * len(col))
        pe_agj = sp.csr_matrix((values, (row, col)), shape=(hg.number_of_nodes('p'), e_num), dtype=np.float64)
        ind_1 = hg.incidence_matrix(typestr='out', etype='tp')._indices()
        row_1, col_1 = ind_1[0], ind_1[1] + hg.number_of_edges('pc') + hg.number_of_edges('pa')
        values_1 = torch.Tensor([1] * len(col_1))
        te_adj = sp.csr_matrix((values_1, (row_1, col_1)), shape=(hg.number_of_nodes('t'), e_num),
                               dtype=np.float64)
        new_hg = dgl.heterograph({
            ('p', 'pe', 'e'): pe_agj,
            ('e', 'ep', 'p'): pe_agj.transpose(),
            ('a', 'ae', 'e'): ae_adj,
            ('e', 'ea', 'a'): ae_adj.transpose(),
            ('c', 'ce', 'e'): ce_adj,
            ('e', 'ec', 'c'): ce_adj.transpose(),
            ('t', 'te', 'e'): te_adj,
            ('e', 'et', 't'): te_adj.transpose()
        })
        print(new_hg)
        return new_hg



def main(config):
    dataloader = load_data(config.data_config)
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    if isinstance(config.data_config['primary_type'], str):
        multi = False
    else:
        multi = True
        print('Classification on types:{}'.format(config.data_config['primary_type']))
    hg = dataloader.heter_graph
    CF_data = dataloader.load_classification_data()
    features, labels, num_classes, train_idx, val_idx, test_idx = CF_data

    train_data = SimPairDataset(CF_data, config.train_config['neg_num_for_each_hop'],
                                config.data_config['primary_type'])
    device = torch.device('cuda:{}'.format(args.n) if torch.cuda.is_available() else 'cpu')

    model = ContextGNN(hg, config.model_config, train_data.num_classes, n_out=labels.max().item() + 1)
    model = model.to(device)
    labels = torch.LongTensor(labels)
    train_idx = torch.LongTensor(train_idx)
    test_idx = torch.LongTensor(test_idx)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    test_idx = test_idx.to(device)

    if config.train_config['continue']:
        model = load_latest_model(config.train_config['checkpoint_path'], model)

    stopper = EarlyStopping(checkpoint_path=config.train_config['checkpoint_path'], config=config,
                            patience=config.train_config['patience'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.train_config['lr'],
                                 weight_decay=config.train_config['l2'])  # torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.train_config['factor'],
                                  patience=config.train_config['patience'] // 3, verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    beta = config.train_config['beta']
    sup_weight = config.train_config['sup_weight']
    unsup_weight = config.train_config['unsup_weight']
    print("Start training...")
    for epoch in range(config.train_config['total_epoch']):
        output_embs = model()
        if multi == True:
            emb_list = [output_embs[t] for t in config.data_config['primary_type']]
            emb = torch.cat(emb_list)
        else:
            emb = output_embs[model.primary_type]
        pos_loss, neg_loss = model.get_loss(emb, train_data, sup_weight, unsup_weight)
        loss = pos_loss + beta * neg_loss
        logits = model.get_predict(emb)
        train_loss = loss_fn(logits[train_idx], labels[train_idx])
        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_idx], labels[train_idx])
        with torch.no_grad():
            val_pos_loss, val_neg_loss = model.get_loss(emb, train_data, sup_weight, unsup_weight, val=True)
            val_cf_loss = loss_fn(logits[val_idx], labels[val_idx])
            val_loss = val_pos_loss + beta * val_neg_loss  # get_loss(emb[val_idx], labels[val_idx])
        val_acc, val_micro_f1, val_macro_f1 = score(logits[val_idx], labels[val_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Pos Loss: {}, Neg Loss: {}".format(pos_loss.item(), neg_loss.item()))
        scheduler.step(val_loss)  # Reduce learning rate
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, train_loss.item(), train_micro_f1, train_macro_f1, val_cf_loss.item(), val_micro_f1, val_macro_f1))
        early_stop = stopper.step(val_loss, model)
        if config.train_config['evaluate_every'] > 0 and epoch % config.train_config['evaluate_every'] == 0:
            if multi == True:
                emb_list = [output_embs[t] for t in config.data_config['primary_type']]
                p_emb = torch.cat(emb_list).detach()
            else:
                p_emb = output_embs[model.primary_type].detach()
            evaluate(p_emb, CF_data, None,
                     dataset=config.data_config["dataset"],
                     method=config.evaluate_config['method'],
                     metric=config.data_config['task'], save_result=False,
                     result_path=config.evaluate_config['result_path'],
                     random_state=config.evaluate_config['random_state'],
                     max_iter=config.evaluate_config['max_iter'], n_jobs=config.evaluate_config['n_jobs'],
                     n_time=config.evaluate_config['n_time'])
        if early_stop:
            break

    checkpoint_path = stopper.filepath
    evaluate_task(config, train_data.num_classes, checkpoint_path)
    return


if __name__ == "__main__":
    import config

    setup_seed(config.evaluate_config['random_state'])
    main(config)
