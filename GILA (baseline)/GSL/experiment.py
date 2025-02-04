import os
import sys
import pickle
from contextlib import redirect_stdout

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

from GSL.data import *
from GSL.model import *
from GSL.model.baselines import *
from GSL.utils import (accuracy, get_knn_graph, macro_f1, micro_f1, bacc,
                       random_add_edge, random_drop_edge, seed_everything,
                       sparse_mx_to_torch_sparse_tensor, feature_mask, apply_feature_mask,
                       split_semi_dataset, get_idx_info,get_dataset)


class Experiment(object):
    def __init__(self, model_name, 
                 dataset, ntrail, 
                 data_path, config_path,
                 sparse: bool = False,
                 metric: list =['acc'],
                 gpu_num: int = 0,
                 use_knn: bool = False,
                 k: int = 5,
                 drop_rate: float = 0.,
                 add_rate: float = 0.,
                 mask_feat_rate: float = 0.,
                 use_mettack: bool = False,
                 ptb_rate: float = 0.05,
                 imb_ratio: int = 1,
                 fake_samples: bool = False,   
                 prompts: bool = False,             
                 seed: int=0
                 ):

        self.eval_metric = {
            'acc': accuracy,
            'macro-f1': macro_f1,
            'micro-f1': micro_f1,
            'bacc': bacc
        }
        self.sparse = sparse
        self.ntrail = ntrail
        self.metric = {m: self.eval_metric[m] for m in metric}
        self.imb_ratio = imb_ratio
        self.seed = seed


        self.config_path = config_path
        self.device = torch.device("cuda:"+str(gpu_num) if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.dataset_name = dataset.lower()
        self.model_dict = {'GCN': GCN_Trainer, 'GAT': GAT_Trainer}
     
        # Load graph datasets
        if self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            self.data = Dataset(fake_samples=fake_samples,prompts=prompts,imb_ratio=imb_ratio, root=data_path, name=self.dataset_name, use_mettack=use_mettack, ptb_rate=ptb_rate)
   
        self.fake_idx = self.data.fake_idx
        self.fake_samples = fake_samples
        self.prompts = prompts

        if isinstance(self.data, Dataset):
            # Modify graph structures
            adj = self.data.adj
            features = self.data.features

            mask = feature_mask(features, mask_feat_rate)
            apply_feature_mask(features, mask)

            # Randomly drop edges
            if drop_rate > 0:
                adj = random_drop_edge(adj, drop_rate)

            # Randomly add edges
            if add_rate > 0:
                adj = random_add_edge(adj, add_rate)

            # Use knn graph instead of the original structure
            if use_knn:
                adj = get_knn_graph(features, k, self.dataset_name)

            # Sparse or notyao
            if not self.sparse:
                self.data.adj = torch.from_numpy(adj.todense()).to(torch.float)
            else:
                self.data.adj = sparse_mx_to_torch_sparse_tensor(adj)
            
        self.data = self.data.to(self.device)
   
        
    def run(self, params=None):
        """
        Run the experiment
        """
        test_results = {m: [] for m in self.metric}
        num_feat, num_class = self.data.num_feat, self.data.num_class
        num_samples = self.data.features.shape[0]

        for i in range(self.ntrail,self.ntrail+1):

            seed_everything(i)

            # Initialize the GSL model
            if self.model_name in ['SLAPS', 'CoGSL', 'HGSL', 'GTN', 'HAN']:
                model = self.model_dict[self.model_name](num_feat, num_class, self.metric,
                                                         self.config_path, self.dataset_name, self.device, self.data) # TODO modify the config according to the search space
            else:
                model = self.model_dict[self.model_name](num_samples, num_feat, num_class, self.metric,
                                                         self.config_path, self.dataset_name, self.device, params, self.prompts, self.fake_idx)
            self.model = model.to(self.device)
            
            # Structure Learning
            self.model.fit(self.data, split_num=i)

            result = self.model.best_result
            for m in self.metric:
                test_results[m].append(result[m])
            print('Run: {} | Test result: {}'.format(i+1, result))

        formatted_results = {key: [float(val.item()) if isinstance(val, torch.Tensor) else float(val) for val in values] for key, values in test_results.items()}

        # TODO: support multiple metrics
        exp_info = '------------------------------------------------------------------------------\n' \
                   'Experimental settings: \n' \
                   'Model: {} | Dataset: {} | Metric: {} \n' \
                   'Result: {} \n' \
                   '------------------------------------------------------------------------------'.\
                   format(self.model_name, self.dataset_name, list(self.metric.keys()), formatted_results)

        print(exp_info)
        return formatted_results
    
    def objective(self, params):
        with redirect_stdout(open(os.devnull, "w")):
            return {'loss': -self.run(params), 'status': 'ok'}

    def hp_search(self, hyperspace_path):
        with open(hyperspace_path, "r") as fin:
            raw_text = fin.read()
            raw_space = edict(yaml.safe_load(raw_text))
        
        space_hyperopt = {} 
        for key, config in raw_space.items():
            if config.type == 'choice':
                space_hyperopt[key] = hp.choice(key, config.values)
            
            elif config.type == 'uniform':
                space_hyperopt[key] = hp.uniform(key, config.bounds[0], config.bounds[1])
            
        print(space_hyperopt)
        trials = Trials()
        best = fmin(self.objective, space_hyperopt, algo=tpe.suggest, max_evals=100, trials=trials)
        print(trials.best_trial)
        
        with open('trails.pkl', 'wb') as f:
            pickle.dump(trials, f)

