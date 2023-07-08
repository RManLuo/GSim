#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 11:11
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : evaluate.py
# @Software: PyCharm
# @Describe:

from models import ContextGNN
from utils import load_data, evaluate, load_latest_model, save_attention_matrix, generate_attention_heat_map, \
    save_config
import torch
import importlib
import os
import argparse
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from utils.helper import setup_seed

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1

def evaluate_task(config, num_classes, checkpoint_path=None):
    setup_seed(config.evaluate_config['random_state'])
    dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    CF_data = dataloader.load_classification_data()
    features, labels, num_classes, train_idx, val_idx, test_idx = CF_data
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    if not checkpoint_path:
        model = ContextGNN(hg, config.model_config, num_classes)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        config_path = os.path.join(checkpoint_path, 'config')
        config_path = os.path.relpath(config_path)
        config_file = config_path.replace(os.sep, '.')
        model_path = os.path.join(checkpoint_path, 'model.pth')
        config = importlib.import_module(config_file)
        model = ContextGNN(hg, config.model_config,num_classes,n_out=labels.max().item() + 1)
        model.load_state_dict(torch.load(model_path))
    model.eval()
    output_embs = model()
    if not isinstance(config.data_config['primary_type'], str):
        emb_list = [output_embs[t] for t in config.data_config['primary_type']]
        emb = torch.cat(emb_list)
    else:
        emb = output_embs[model.primary_type]
    logits = model.get_predict(emb)
    train_acc, train_micro_f1, train_macro_f1 = score(logits[train_idx], labels[train_idx])
    val_acc, val_micro_f1, val_macro_f1 = score(logits[val_idx], labels[val_idx])
    test_acc, test_micro_f1, test_macro_f1 = score(logits[test_idx], labels[test_idx])
    print('Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
          'Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        train_micro_f1, train_macro_f1, val_micro_f1, val_macro_f1))
    print('Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(test_micro_f1, test_macro_f1))
    if not isinstance(config.data_config['primary_type'], str):
        emb_list = [output_embs[t] for t in config.data_config['primary_type']]
        p_emb = torch.cat(emb_list).detach()
    else:
        p_emb = output_embs[model.primary_type].detach()
    # LP_data = dataloader.load_links_prediction_data()
    result_save_path = evaluate(p_emb, CF_data, None, dataset=config.data_config["dataset"], method=config.evaluate_config['method'],
                                metric=config.data_config['task'], save_result=True,
                                result_path=config.evaluate_config['result_path'],
                                random_state=config.evaluate_config['random_state'],
                                max_iter=config.evaluate_config['max_iter'], n_jobs=config.evaluate_config['n_jobs'],
                                n_time=config.evaluate_config['n_time'],
                                sim_matrix=config.evaluate_config['use_sim_matrix'])
    if result_save_path:
        save_config(config, result_save_path)
        model_save_path = os.path.join(result_save_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        if config.evaluate_config['save_heat_map']:
            attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['K_length'])
            if attention_matrix_path:
                generate_attention_heat_map(hg.ntypes, attention_matrix_path)


if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(description='Which checkpoint to load?')
    parser.add_argument('-path', default=None, type=str, help='checkpoint path')
    args = parser.parse_args()
    evaluate_task(config, args.path)
