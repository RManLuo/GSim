# Gsim

----

Official code implementation for TKDE paper GSim: A Graph Neural Network Based Relevance Measure for Heterogeneous Graphs

> L. Luo, Y. Fang, M. Lu, X. Cao, X. Zhang and W. Zhang, "GSim: A Graph Neural Network Based Relevance Measure for Heterogeneous Graphs," in IEEE Transactions on Knowledge and Data Engineering, doi: https://doi.org/10.1109/TKDE.2023.3271425.

## Environments

python == 3.6

CPU: I7-8700K 

RAM: 64GB

GPU: RTX 3080 

CUDA: 10.1

## Requirements

```bash
torch==1.6.0
torch-cluster==1.5.7
torch-scatter==2.0.5
torch-sparse==0.6.7
torch-spline-conv==1.2.0
torch_geometric==1.6.1
matplotlib==2.2.3
networkx==2.4
dgl==0.4.3.post2
numpy==1.16.6
scipy==1.4.1
scikit-learn==0.24.2
pyclustering == 0.10.1.2
```

## Config

```
vi config.py
```
config.py example

```python
import os

config_path = os.path.dirname(__file__)
data_config = {
    'data_path': os.path.join(config_path, 'data'),
    'dataset': 'DBLP',
    'data_name': 'DBLP.mat',
    'primary_type': 'a',
    'query_type': 'a', # for recall evaluation
    'return_type': 'a', # for recall evaluation
    'query_name': '', # for recall evaluation
    'task': ['CL'],
    'K_length': 2,
    'resample': False,
    'random_seed': 123,
    'train_val_test_ratio': [0.25, 0.25, 0.5]
}

model_config = {
    'primary_type': data_config['primary_type'],
    'auxiliary_embedding': 'linear',  # auxiliary embedding generating method: non_linear, linear, embedding
    'K_length': data_config['K_length'],
    'embedding_dim': 128,
    'in_dim': 128,
    'out_dim': 128,
    'num_heads': 2,
    'merge': 'mean',  # Multi head Attention merge method: linear, mean, stack
    'g_agg_type': 'mean',  # Graph representation encoder: mean, sum
    'drop_out': 0.3,
    'cgnn_non_linear': True,  # Enable non linear activation function for CGNN
    'multi_attn_linear': False,  # Enable atten K/Q-linear for each type
    'graph_attention': True,
    'kq_linear_out_dim': 128,
    'path_attention': False,  # Enable Context path attention
    'c_linear_out_dim': 8,
    'enable_bilinear': False,  # Enable Bilinear for context attention
    'enable_relational_passing': True,  # Enable Relational Message Passing
    'gru': True,
    'add_init': False,
    'max_norm': 2,  # Renorm, 0 for disable.
}

train_config = {
    'continue': False,
    'evaluate_every': -1,
    'beta': 1,
    'sup_weight': 1,
    'unsup_weight': 1,
    'lr': 0.001,
    'l2': 0.1,
    'factor': 0.2,
    'total_epoch': 10000000,
    'batch_size': 1024 * 20,
    'pos_num_for_each_hop': [20, 20, 20, 20, 20, 20, 20, 20, 20],
    'neg_num_for_each_hop': -1,
    'sample_workers': 8,
    'patience': 100,
    'checkpoint_path': os.path.join(config_path, 'checkpoint', data_config['dataset'])
}

evaluate_config = {
    'method': 'SpectralClustering',  # 'kmeans', 'kmedoids', 'AgglomerativeClustering', 'SpectralClustering'
    'use_sim_matrix': True,
    'save_heat_map': False,
    'result_path': os.path.join('result', data_config['dataset']),
    'random_state': 123,
    'max_iter': 500,
    'n_jobs': 1,
    'n_time': 100,
    'eval_time': 50,
    'K': 10 # number of results for recall exp.
}
```

## Train and Evaluate
``` bash
python3 main.py
```

## BibTex
```tex
@ARTICLE{10111040,
  author={Luo, Linhao and Fang, Yixiang and Lu, Moli and Cao, Xin and Zhang, Xiaofeng and Zhang, Wenjie},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={GSim: A Graph Neural Network Based Relevance Measure for Heterogeneous Graphs}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TKDE.2023.3271425}}
```
