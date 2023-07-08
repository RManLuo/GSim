#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 23:25
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : evaluator.py
# @Software: PyCharm
# @Describe:
from scipy.linalg import block_diag
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AffinityPropagation, AgglomerativeClustering
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.metrics import f1_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, \
    silhouette_score, cluster, completeness_score, pair_confusion_matrix
from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import os
import json
import torch
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from .preprocess import SimPairDataset
import matplotlib

matplotlib.use('module://backend_interagg')


def normalize_sim(sim_matrix):
    norm_sim = np.identity(len(sim_matrix))
    norm_sim = torch.Tensor(norm_sim)
    for i in range(len(sim_matrix)):
        for j in range(i, len(sim_matrix)):
            sum_row = torch.sum(sim_matrix[i, :])
            sum_col = torch.sum(sim_matrix[:, j])
            if (sum_row + sum_col) == 0:
                norm_sim[i, j] = 0
                norm_sim[j, i] = 0
            else:
                norm_sim[i, j] = sim_matrix[i, j] / (sum_row + sum_col)
                norm_sim[j, i] = norm_sim[i, j]
    return norm_sim


def cosine_sim(emb):
    sim_matrix = torch.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    sim_matrix = (1 + sim_matrix) / 2
    return sim_matrix


def get_simiarity_matrix(emb):
    n = torch.norm(emb, p=2, dim=1)
    sim_matrix = torch.sigmoid(emb @ emb.t())
    return sim_matrix.detach().cpu().numpy()


def KNN_train(x, y):
    knn = KNeighborsClassifier()
    knn.fit(x, y)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class Evaluator(object):
    def __init__(self, method, CF_data, LP_data, dataset, result_path='./result', random_state=123, max_iter=150,
                 n_jobs=1, sim_matrix=True, test_only=True):
        self.method = method
        self.CF_data = CF_data
        self.LP_data = LP_data
        self.result_path = result_path
        self.sim_matrix = sim_matrix
        if not os.path.exists(self.result_path):
            os.makedirs((self.result_path))
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.result = {}
        self.dataset = dataset
        self.test_only = test_only

    def get_model(self):
        if self.method == "KNN":
            model = KNeighborsClassifier()
        elif self.method == "LR":
            model = LogisticRegression(solver='lbfgs', random_state=self.random_state, max_iter=self.max_iter,
                                       n_jobs=self.n_jobs, multi_class='auto')
        elif self.method == "SVM":
            model = SVC()
        return model

    def evluate_CF(self, emb_feature):
        features, labels, num_classes, train_idx, test_idx = self.CF_data
        model = self.get_model()
        model.fit(emb_feature[train_idx], labels[train_idx])
        score = model.predict(emb_feature[test_idx])  #
        micro_f1 = f1_score(labels[test_idx], score, average='micro')
        macro_f1 = f1_score(labels[test_idx], score, average='macro')
        self.result['CF'] = {'Micro f1': micro_f1, 'Macro f1': macro_f1}
        print("Node classification result: ")
        print('Micro f1: ', micro_f1)
        print('Macro f1: ', macro_f1)

    def evluate_LP(self, emb_feature):
        features, src_train, src_test, dst_train, dst_test, labels_train, labels_test = self.LP_data
        train_edges_feature = self._concat_edges_feture(emb_feature, src_train, dst_train)
        test_edges_feature = self._concat_edges_feture(emb_feature, src_test, dst_test)
        model = self.get_model()
        model.fit(train_edges_feature, labels_train)
        score = model.predict(test_edges_feature)
        f1 = f1_score(labels_test, score)
        auc_score = roc_auc_score(labels_test, score)
        self.result['LP'] = {'AUC': auc_score, 'F1': f1}
        print("Link Prediction result: ")
        print('AUC: ', auc_score)
        print('F1: ', f1)

    def evluate_CL(self, emb_feature, time=10):
        features, labels, num_classes, train_idx, val_idx, test_idx = self.CF_data
        if self.test_only:
            x_idx = test_idx
        else:
            x_idx = np.concatenate((train_idx, test_idx))
        x = emb_feature[x_idx]
        sim_matrix = 0
        # Enable Sim_matrix
        if self.sim_matrix:
            sim_matrix = get_simiarity_matrix(x)
        if self.method in ['kmeans', 'kmedoids', 'AgglomerativeClustering', 'SpectralClustering']:
            method = self.method
        else:
            method = "kmeans"

        if method == "kmeans":
            estimator = KMeans(
                n_clusters=num_classes,
                max_iter=self.max_iter)
        elif method == "kmedoids":
            if self.sim_matrix:
                estimator = KMedoids(n_clusters=num_classes, metric='precomputed', max_iter=self.max_iter)
            else:
                estimator = KMedoids(n_clusters=num_classes, max_iter=self.max_iter)
        elif method == 'AgglomerativeClustering':
            if self.sim_matrix:
                estimator = AgglomerativeClustering(n_clusters=num_classes, affinity="precomputed", linkage='average')
            else:
                estimator = AgglomerativeClustering(n_clusters=num_classes, linkage='average')
        elif method == 'SpectralClustering':
            if self.sim_matrix:
                estimator = SpectralClustering(n_clusters=num_classes, affinity="precomputed", random_state=123)
            else:
                estimator = SpectralClustering(n_clusters=num_classes, random_state=123)

        x = x.cpu().numpy()
        y = labels[x_idx]

        ARI_list = []
        NMI_list = []
        Purity_list = []
        silhouette_score_list = []
        f_list = []
        cp_list = []
        for i in range(time):

            if self.sim_matrix:
                if method == "kmeans":
                    # def user_function(point1, point2):
                    #     return 1 / (1 + np.exp(-np.sum(point1 * point2)))

                    user_function = lambda point1, point2: 1 - (1 / (1 + np.exp(-np.sum(point1 * point2))))
                    metric = distance_metric(type_metric.USER_DEFINED, func=user_function)
                    initial_medoids = kmeans_plusplus_initializer(x, num_classes).initialize()
                    estimator = kmeans(x, initial_medoids, metric=metric)
                    estimator.process()
                    clusters = estimator.get_clusters()
                    y_pred = self._cluster_to_predict_label(clusters)
                elif method in ["kmedoids", "AgglomerativeClustering"]:
                    y_pred = estimator.fit_predict(1 - sim_matrix)
                elif method == "SpectralClustering":
                    y_pred = estimator.fit_predict(sim_matrix)
            else:
                y_pred = estimator.fit_predict(x)
            score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
            NMI_list.append(score)
            s2 = adjusted_rand_score(y, y_pred)
            ARI_list.append(s2)
            # silhouette_score
            s3 = silhouette_score(x, y_pred, metric='euclidean')
            silhouette_score_list.append(s3)
            s4 = purity_score(y, y_pred)
            Purity_list.append(s4)
            f_beta = self.get_f_measure(y, y_pred)
            f_list.append(f_beta)
            cp_list.append(completeness_score(y, y_pred))
        # print('NMI_list: {}'.format(NMI_list))
        score = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        s3 = sum(silhouette_score_list) / len(silhouette_score_list)
        s4 = sum(Purity_list) / len(Purity_list)
        s5 = sum(f_list) / len(f_list)
        s7 = sum(cp_list) / len(cp_list)

        print(
            'AVG: {}, F-score: {:.4f}, NMI: {:.4f} , ARI: {:.4f}, Purity: {:.4f}, silhouette: {:.4f}, completeness score: {:.4f}'.format(
                time, s5, score, s2, s4, s3, s7))

        self.result['CL'] = {'F-score': s5, 'NMI': score, 'ARI': s2, 'Purity': s4, 'silhouette': s3,
                             'completeness score': s7, 'sim_matrix': sim_matrix}

    def _cluster_to_predict_label(self, cluster):
        point_idxs = []
        y_pred = []
        for y, x in enumerate(cluster):
            point_idxs.extend(x)
            y_pred += [y] * len(x)
        result_list = [i for _, i in sorted(zip(point_idxs, y_pred))]
        return result_list

    def get_f_measure(self, labels_true, labels_pred, beta=1.):
        (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
        p, r = tp / (tp + fp), tp / (tp + fn)
        f_beta = (1 + beta ** 2) * (p * r / ((beta ** 2) * p + r))
        return f_beta

    def _concat_edges_feture(self, emb_feature, src_list, dst_list):
        src_feature = emb_feature[src_list]
        dst_feature = emb_feature[dst_list]
        edges_feature = src_feature * dst_feature
        # edges_feature = np.concatenate([src_feature, dst_feature], 1)
        return edges_feature

    def dump_result(self, p_emb, metric):
        dir_name = ''
        if 'CF' in metric:
            dir_name += "CF_{:.2f}_{:.2f}_".format(self.result['CF']['Micro f1'], self.result['CF']['Macro f1'])
        if 'LP' in metric:
            dir_name += "LP_{:.2f}_{:.2f}_".format(self.result['LP']['AUC'], self.result['LP']['F1'])
        if 'CL' in metric:
            dir_name += "CL_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(self.result['CL']['NMI'], self.result['CL']['ARI'],
                                                                self.result['CL']['Purity'],
                                                                self.result['CL']['silhouette'])
            if self.sim_matrix:
                sim_matrix = self.result['CL'].pop('sim_matrix')
        model_path = os.path.join(self.result_path, dir_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        with open(os.path.join(model_path, 'result.json'), 'w') as f:
            json.dump(self.result, f)
        np.save(os.path.join(model_path, 'p_emb.npy'), p_emb)
        # Plot cluster
        features, labels, num_classes, train_idx, val_idx, test_idx = self.CF_data
        # plot Emb
        x_idx = np.concatenate((train_idx, test_idx))
        x = p_emb[x_idx]
        y = labels[x_idx]
        tsne = TSNE()
        tsne.fit_transform(x)  # 降维
        plt.scatter(
            x[:, 0],
            x[:, 1],
            c=y,
        )
        plt.savefig(os.path.join(model_path, 'emb_tsne.jpg'))
        # Plot SIM
        if self.sim_matrix:
            np.save(os.path.join(model_path, 'sim_matrix.npy'), sim_matrix)
            sim = plt.matshow(sim_matrix, interpolation='nearest', cmap="Blues")
            plt.colorbar(sim)
            plt.savefig(os.path.join(model_path, 'raw_sim_matrix.jpg'))

            # Group by label
            if not self.test_only:
                test_idx = np.concatenate((train_idx, test_idx))
            test_labels = labels[test_idx]
            test_idx = [i for i in range(len(test_labels))]
            df_group = pd.DataFrame({'idx': test_idx, 'label': test_labels}).groupby('label')
            test_idxs_list = []
            labels_list = []
            for key, group in df_group:
                pos_idxs = group['idx'].values
                test_idxs_list.append(pos_idxs)
                labels_list.append(key)
            block = []
            for x in test_idxs_list:
                tmp = []
                for y in test_idxs_list:
                    tmp.append(sim_matrix.take(x, 0).take(y, 1))
                block.append(tmp)
            group_sim = np.block(block)
            plt.figure(figsize=(11,8))
            block_sim = plt.matshow(group_sim, interpolation='nearest', cmap="Blues",fignum=0)
            split_loc = 0
            label_loc = []
            plt.title(self.dataset+' - '+ os.path.basename(os.getcwd()), y=-0.1, fontsize=20)
            if self.dataset == 'ACM':
                labels_map = {0: 'DM', 1: 'DB', 2: 'CM'}
            elif self.dataset == 'DBLP':
                labels_map = {0: 'DB', 1: 'DM', 2: 'AI', 3: 'IR'}
            elif self.dataset == 'IMDB':
                labels_map = {0: 'Action', 1: 'Adventure', 2: 'Drama'}
            else:
                labels_map = {v: 'G-' + str(v) for v in labels_list}
            for i in range(len(test_idxs_list)):
                num = len(test_idxs_list[i])
                # Set grid
                if i < len(test_idxs_list) - 1:
                    plt.axhline(y=split_loc + num - 0.5, c="#D03F0C", lw=2)
                    plt.axvline(x=split_loc + num - 0.5, c="#D03F0C", lw=2)
                label_loc.append(split_loc + num / 2)
                split_loc += num
            # Set label
            plt.xticks(label_loc, [labels_map[v] for v in labels_list], fontsize=20)
            plt.yticks(label_loc, [labels_map[v] for v in labels_list], fontsize=20)
            plt.colorbar(block_sim)
            plt.savefig(os.path.join(model_path, 'group_sim_matrix.jpg'))
            plt.show()

        print("Result save in {}".format(model_path))
        return model_path





class recall_evaluator(object):
    def __init__(self, config, data, CF_data):
        # set the mode of experiment
        self.is_visual_sim = False
        self.is_recall_top_k = False
        self.is_cal_avg_rec = False
        self.primary_type = config.data_config['primary_type']
        self.query_type = config.data_config['query_type']
        self.return_type = config.data_config['return_type']
        self.config = config
        features, labels, num_classes, train_idx, val_idx, test_idx = CF_data
        self.labels = labels
        self.test_idx = test_idx
        if isinstance(self.primary_type, list):
            self.is_multi = True
            num_a = 14373
            num_p = 14298
            num_c = 20
            num_a_sim = 2029
            num_p_sim = 50
            num_c_sim = 10
            self.start_idx = {'a':0, 'p':num_a, 'c':num_a+num_p}
            self.end_idx = {'a':num_a, 'p':num_a+num_p, 'c':num_a+num_p+num_c}
            self.start_idx_sim = {'a':0, 'p':num_a_sim, 'c':num_a_sim+num_p_sim}
            self.end_idx_sim = {'a': num_a_sim, 'p': num_a_sim+num_p_sim, 'c': num_a_sim + num_p_sim + num_c_sim}
        else:
            self.is_multi = False

        self.dataset = config.data_config['dataset']
        if self.dataset == 'ACM':
            self.labels_map = {0: 'DM', 1: 'DB', 2: 'CM'}
        elif self.dataset == 'DBLP':
            self.labels_map = {0: 'DB', 1: 'DM', 2: 'AI', 3: 'IR'}
        elif self.dataset == 'AIFB':
            test_label = labels[test_idx]
            labels_list = list(set(test_label))
            self.labels_map = {v: 'G-' + str(v) for v in labels_list}
        else:
            self.labels_map = {0: 'Action', 1: 'Adventure', 2: 'Drama'}

        if config.data_config['dataset'] == 'ACM':
            self.query_type = self.query_type.upper()
            if self.query_type == 'S': self.query_type = 'L'
            self.return_type = self.return_type.upper()
            if self.return_type == 'S': self.return_type = 'L'
            self.name_list = np.array([i[0][0] for i in data[self.query_type]])
            self.res_name_list = np.array([i[0][0] for i in data[self.return_type]])
        elif config.data_config['dataset'] == 'IMDB':
            self.name_list = np.array(self.load_name('IMDB', self.query_type))
            self.res_name_list = np.array(self.load_name('IMDB', self.return_type))
        elif config.data_config['dataset'] == 'DBLP':
            self.name_list = np.array(self.load_name('DBLP', self.query_type))
            self.res_name_list = np.array(self.load_name('DBLP', self.return_type))
        else:
            pass


    def load_name(self, dataset, type):
        name_list = []
        if dataset == 'IMDB':
            if type == 'm':
                file_path = os.path.join(self.config.data_config['data_path'], self.config.data_config['dataset'],
                                         'movie_index.txt')
                with open(file_path) as f:
                    for l in f.readlines():
                        l = l.replace("\n", "")
                        idx, name = l.split(",")
                        name = name.strip().replace(u'\xa0', u'')
                        name_list.append(name)
            else:
                print('No such function to deal with type:{} yet'.format(type))
        elif dataset == 'DBLP':
            if type == 'c':
                file_path = os.path.join(self.config.data_config['data_path'], self.config.data_config['dataset'],
                                         'conf.txt')
            elif type == 'p':
                file_path = os.path.join(self.config.data_config['data_path'], self.config.data_config['dataset'],
                                         'paper.txt')
                # process it seperately due to some errors
                with open(file_path, encoding='gb2312') as f:
                    for l in f.readlines():
                        # l = l.replace("\n", "")
                        idx, name = l.split("\t")
                        name = name.strip().replace(u'\xa0', u'')
                        name_list.append(name)
                return name_list
            elif type == 'a':
                file_path = os.path.join(self.config.data_config['data_path'], self.config.data_config['dataset'],
                                         'author.txt')
            else:
                print('No such function to deal with type:{} yet'.format(type))
            with open(file_path) as f:
                for l in f.readlines():
                    # l = l.replace("\n", "")
                    idx, name = l.split("\t")
                    name = name.strip().replace(u'\xa0', u'')
                    name_list.append(name)

        # for i in set(self.labels):
        #     idx = np.where(self.labels[self.test_idx] == i)
        #     print('class {}:{}'.format(self.labels_map[i], np.array(name_list)[self.test_idx][idx][:20]))
        # print(np.array(name_list)[self.test_idx])
        # print(self.labels[self.test_idx])
        return name_list


    def select_mode(self):
        if isinstance(self.config.data_config['query_name'], list):
            self.is_visual_sim = True
            query_id_list = []
            for query_name in self.config.data_config['query_name']:
                query_id_list.append(np.where(self.name_list == query_name)[0][0])
            if self.is_multi:
                query_id_list = query_id_list + np.repeat(self.start_idx[self.query_type], len(query_id_list))
            print('-----------------Visualization Experiment / similarity matrix-----------------')
            print('query node list:{}'.format(self.config.data_config['query_name']))
        elif self.config.data_config['query_name'] != '':
            self.is_recall_top_k = True
            query_id_list = np.where(self.name_list == self.config.data_config['query_name'])[0]
            print('-----------------Visualization Experiment / recall top-k -----------------')
            if self.is_multi:
                query_id_list = query_id_list + self.start_idx[self.query_type]
                print('query node :{}, label:{}'.format(self.config.data_config['query_name'],
                                                        self.labels_map[self.labels[query_id_list[0]]]))
            else:
                print('query node :{}, label:{}'.format(self.config.data_config['query_name'],
                                                        self.labels_map[self.labels[query_id_list[0]]]))
        else:
            self.is_cal_avg_rec = True
            query_id_list = np.random.choice(self.test_idx, size=self.config.evaluate_config['eval_time'])
            print('-----------------Query Experiment / average recall-----------------')
            if self.is_multi:
                print('query nodes from type: {}'.format(self.config.data_config['primary_type']))
            else:
                print('query nodes from type: {}'.format(self.config.data_config['query_type']))
        print('query_id_list:{}'.format(query_id_list))
        self.query_id_list = query_id_list
        return query_id_list

    def eval(self, sim_matrix, save=True):
        # Evaluate
        query_sim_matrix = np.zeros((len(self.query_id_list), len(self.query_id_list)))
        if self.is_visual_sim:
            # sort by labels
            labels_dict = {i: self.labels[i] for i in self.query_id_list}
            labels_sorted = sorted(labels_dict.items(), key=lambda x: x[1])
            query_id_list = [i[0] for i in labels_sorted]
            labels_list = [i[1] for i in labels_sorted]
            num_class_list = [labels_list.count(label) for label in set(labels_list)]
            for i in range(len(query_id_list)):
                for j in range(len(query_id_list)):
                    id_in_sim_i = np.where(self.test_idx == query_id_list[i])[0][0]
                    id_in_sim_j = np.where(self.test_idx == query_id_list[j])[0][0]
                    query_sim_matrix[i, j] = sim_matrix[id_in_sim_i, id_in_sim_j]

            # visualization
            plt.figure(figsize=(12, 10))
            sim = plt.matshow(query_sim_matrix, interpolation='nearest', cmap="Blues", fignum=0)
            plt.colorbar(sim)
            plt.title(self.config.data_config['dataset'] + ' - ' + os.path.basename(os.getcwd()), y=-0.1, fontsize=12)
            split_loc = 0
            min_sim = np.min(query_sim_matrix)
            max_sim = np.max(query_sim_matrix)
            for i in range(len(query_id_list)):
                for j in range(len(query_id_list)):
                    if query_sim_matrix[i, j] > 0.5*(max_sim-min_sim)+min_sim:
                        plt.annotate(np.round(query_sim_matrix[i, j], 2), xy=(i, j), xytext=(i - 0.25, j), color='w')
                    else:
                        plt.annotate(np.round(query_sim_matrix[i, j], 2), xy=(i, j), xytext=(i - 0.25, j))
            for i in range(len(set(labels_list))):
                num = num_class_list[i]
                # Set grid
                if i < len(set(labels_list)):
                    plt.axhline(y=split_loc + num - 0.5, c="#D03F0C", lw=2)
                    plt.axvline(x=split_loc + num - 0.5, c="#D03F0C", lw=2)
                split_loc += num
            # Set label
            if self.is_multi:
                plt.xticks(range(len(query_id_list)), [self.name_list[i - self.start_idx[self.query_type]] for i in query_id_list], fontsize=12, rotation=45)
                plt.yticks(range(len(query_id_list)), [self.name_list[i - self.start_idx[self.query_type]] for i in query_id_list], fontsize=12, rotation=45)
            else:
                plt.xticks(range(len(query_id_list)),[self.name_list[i] for i in query_id_list], fontsize=12,rotation=45)
                plt.yticks(range(len(query_id_list)),[self.name_list[i] for i in query_id_list], fontsize=12,rotation=45)
            plt.show()

            if save:
                result_path = os.path.join(self.config.evaluate_config['result_path'], 'visual_sim')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                np.save(os.path.join(result_path, 'query_sim_matrix.npy'), query_sim_matrix)
                plt.savefig(os.path.join(result_path, 'query_sim_matrix.jpg'))
                plt.show()


        else:
            recall_list = []

            if self.is_multi == False:
                for id in self.query_id_list:
                    id_in_sim = np.where(self.test_idx == id)[0][0]
                    res = sim_matrix[id_in_sim, :]
                    sorted_res = torch.sort(res, descending=True)
                    sorted_idx = self.test_idx[sorted_res.indices]

                    if self.is_cal_avg_rec:
                        # idx_K = sorted_res.indices[:self.config.evaluate_config['K']]
                        idx_K = sorted_idx[:self.config.evaluate_config['K']]
                        target_labels = np.repeat(self.labels[id], self.config.evaluate_config['K'])
                        recall_labels = self.labels[idx_K]
                        recall = np.sum(target_labels == recall_labels) / self.config.evaluate_config['K']
                        recall_list.append(recall)
                    elif self.is_recall_top_k:  # visualization experiment
                        for i in range(self.config.evaluate_config['K']):
                            # idx = sorted_res.indices[i]
                            idx = sorted_idx[i]
                            value = sorted_res.values[i]
                            print('{:25}\t{}\t{}'.format(self.res_name_list[idx], value, self.labels_map[self.labels[idx]]))

                if self.is_cal_avg_rec:
                    print('avg_recall: {:.5f}'.format(np.mean(recall_list)))

            else:# for multi-type
                if self.is_cal_avg_rec:
                    for id in self.query_id_list:
                        if id >= self.start_idx['c']: # type c, skip
                            continue
                        id_in_sim = np.where(self.test_idx == id)[0][0]

                        if id < self.start_idx['p']: # type a
                            res = sim_matrix[id_in_sim, self.start_idx_sim['a']:self.end_idx_sim['a']]
                            sorted_res = torch.sort(res, descending=True)
                            sorted_idx = self.test_idx[sorted_res.indices]
                        # elif id >= self.start_idx['c']: # type c
                        #     res = sim_matrix[id_in_sim, self.start_idx_sim['c']:self.end_idx_sim['c']]
                        #     sorted_res = torch.sort(res, descending=True)
                        #     sorted_idx = self.test_idx[sorted_res.indices+self.start_idx_sim['c']]
                        else: # type p
                            res = sim_matrix[id_in_sim, self.start_idx_sim['p']:self.end_idx_sim['p']]
                            sorted_res = torch.sort(res, descending=True)
                            sorted_idx = self.test_idx[sorted_res.indices+self.start_idx_sim['p']]

                        idx_K = sorted_idx[:self.config.evaluate_config['K']]
                        target_labels = np.repeat(self.labels[id], self.config.evaluate_config['K'])
                        recall_labels = self.labels[idx_K]
                        recall = np.sum(target_labels == recall_labels) / self.config.evaluate_config['K']
                        recall_list.append(recall)

                    print('avg_recall: {:.5f}'.format(np.mean(recall_list)))

                else: # is_recall_top_k
                    for id in self.query_id_list:
                        id_in_sim = np.where(self.test_idx == id)[0][0]
                        res = sim_matrix[id_in_sim, self.start_idx_sim[self.return_type]:self.end_idx_sim[self.return_type]]
                        sorted_res = torch.sort(res, descending=True)
                        sorted_idx = self.test_idx[sorted_res.indices + self.start_idx_sim[self.return_type]]

                        for i in range(self.config.evaluate_config['K']):
                            idx = sorted_idx[i]
                            value = sorted_res.values[i]
                            print('{:25}\t{:.4f}\t{}'.format(self.res_name_list[idx - self.start_idx[self.return_type]], value, self.labels_map[self.labels[idx]]))