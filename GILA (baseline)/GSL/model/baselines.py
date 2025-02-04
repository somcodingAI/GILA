from GSL.model.base_gsl import BaseModel
from GSL.model.gcn import GCN
from GSL.model.gat import GAT

from GSL.utils import *
import scipy.sparse as sp

import torch
import torch.nn.functional as F
import torch_sparse

class GCN_Trainer(BaseModel):
    def __init__(self, num_samples, num_features, num_classes, metric, config_path, dataset_name, device, params, prompts, fake_idx):        
        super(GCN_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)

        self.gcn = GCN(num_samples=num_samples,
            in_channels=num_features,
                       hidden_channels=self.config.hidden,
                       out_channels=num_classes,
                       num_layers=2,
                       dropout=self.config.dropout,
                       dropout_adj=0,
                       sparse=self.config.sparse,
                       device=device)
        
    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        features = row_normalize_features(features)
        adj = row_normalize_features(adj)
        
        optimizer = torch.optim.Adam(self.gcn.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        zero_matrix=None 
   
        for epoch in range(self.config.epochs):
            self.gcn.train()
            optimizer.zero_grad()

            output = self.gcn(features, adj, zero_matrix)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.gcn.eval()
                output = self.gcn(features, adj,zero_matrix=None)

                train_results = {m: self.metric[m](output[train_mask], labels[train_mask]) for m in self.metric}
                val_results = {m: self.metric[m](output[val_mask], labels[val_mask]) for m in self.metric}
                test_results = {m: self.metric[m](output[test_mask], labels[test_mask]) for m in self.metric}
            
            results = [f"{m}: Train {100 * train_results[m]:.2f}%, Valid {100 * val_results[m]:.2f}%, Test {100 * test_results[m]:.2f}%" for m in self.metric]
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, " + " | ".join(results))

            self.best_val = {m: float('-inf') for m in self.metric}
            self.best_result = {m: 0.0 for m in self.metric}

            for m in self.metric:
                if val_results[m] > self.best_val[m]: 
                    self.best_val[m] = val_results[m]
                    self.best_result[m] = test_results[m]

        best_results = [f"{m}: {100 * self.best_result[m]:.2f}%" for m in self.metric]
        print('Best Test Results: ', " | ".join(best_results))

class GAT_Trainer(BaseModel):
    def __init__(self, num_samples, num_features, num_classes, metric, config_path, dataset_name, device, params, prompts, fake_id):
        super(GAT_Trainer, self).__init__(num_features, num_classes, metric, config_path, dataset_name, device, params)

        self.gat = GAT(nfeat=num_features,
                       nhid=self.config.hidden,
                       nclass=num_classes,
                       dropout=self.config.dropout,
                       alpha=self.config.alpha,
                       nheads=self.config.nheads,
                       sparse=self.config.sparse
                       )

    def fit(self, dataset, split_num=0):
        adj, features, labels = dataset.adj.clone(), dataset.features.clone(), dataset.labels
        if dataset.name in ['cornell', 'texas', 'wisconsin', 'actor']:
            train_mask = dataset.train_masks[split_num % 10]
            val_mask = dataset.val_masks[split_num % 10]
            test_mask = dataset.test_masks[split_num % 10]
        else:
            train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, dataset.test_mask

        features = row_normalize_features(features)
        adj = row_normalize_features(adj)
        
        optimizer = torch.optim.Adam(self.gat.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        for epoch in range(self.config.epochs):
            self.gat.train()
            optimizer.zero_grad()

            output = self.gat(features, adj)
            loss = F.cross_entropy(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.gat.eval()
                output = self.gat(features, adj)

                train_results = {m: self.metric[m](output[train_mask], labels[train_mask]) for m in self.metric}
                val_results = {m: self.metric[m](output[val_mask], labels[val_mask]) for m in self.metric}
                test_results = {m: self.metric[m](output[test_mask], labels[test_mask]) for m in self.metric}
            
            results = [f"{m}: Train {100 * train_results[m]:.2f}%, Valid {100 * val_results[m]:.2f}%, Test {100 * test_results[m]:.2f}%" for m in self.metric]
            print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, " + " | ".join(results))

            self.best_val = {m: float('-inf') for m in self.metric}
            self.best_result = {m: 0.0 for m in self.metric}

            for m in self.metric:
                if val_results[m] > self.best_val[m]: 
                    self.best_val[m] = val_results[m]
                    self.best_result[m] = test_results[m]

        best_results = [f"{m}: {100 * self.best_result[m]:.2f}%" for m in self.metric]
        print('Best Test Results: ', " | ".join(best_results))
