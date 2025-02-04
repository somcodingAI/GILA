# https://github.com/tkipf/gcn/blob/master/gcn/utils.py
import os
import os.path as osp

import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
import torch


from ..generic_utils import *
from ..constants import VERY_SMALL_NUMBER


from torch_scatter import scatter_add
def split_semi_dataset(total_node, n_data, n_cls, class_num_list, idx_info, device, edge_index):


    new_idx_info = []
    _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool, device=device)
    for i in range(n_cls):
        if n_data[i] > class_num_list[i]:

            cls_idx = torch.randperm(len(idx_info[i]))
            cls_idx = idx_info[i][cls_idx]
            cls_idx = cls_idx[:class_num_list[i]]
            new_idx_info.append(cls_idx)
        else:

            new_idx_info.append(idx_info[i])
        _train_mask[new_idx_info[i]] = True

    assert _train_mask.sum().long() == sum(class_num_list)
    assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

    excluded_nodes = []
    for i in range(n_cls):
        result =np.setdiff1d(idx_info[i].cpu(),new_idx_info[i].cpu())
        if result.size != 0:
            excluded_nodes.append(result)
            
    excluded_nodes= np.concatenate(excluded_nodes, axis=0)
    excluded_nodes = torch.from_numpy(excluded_nodes)

    node_mask = torch.zeros(total_node, dtype=torch.bool, device=device)
    node_mask[excluded_nodes] = True
    row, col = edge_index[0], edge_index[1]

    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask
    
    return _train_mask, new_idx_info, edge_mask


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []

    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_data(args, data_dir, dataset_str, knn_size=None, epsilon=None, knn_metric='cosine', prob_del_edge=None, prob_add_edge=None, seed=1234, sparse_init_adj=False):
    fake_samples=args['fake_samples']
    helper_ratio=args['helper_ratio']
    imb_ratio=args['imb_ratio']
    mix_node_init = args['mix_node_init']
    
    assert (knn_size is None) or (epsilon is None)
    

    if dataset_str =='cora':
        dataset = 'Cora'
    elif dataset_str =='pubmed':
        dataset = 'PubMed'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
    dataset = get_dataset(dataset, path, split_type='public')
    data = dataset[0]
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)        
    
    raw_features = data.x

    edge_index = data.edge_index
    features = normalize_features(raw_features)

    raw_features = torch.Tensor(raw_features)
    features = torch.Tensor(features)
    


    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    
    labels = torch.LongTensor(data.y)

    ######## Data statistic ##
    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
    stats = data.y[data_train_mask]
    n_cls = data.y.max().item() + 1
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(data.y, n_cls, data_train_mask)
    class_num_list = n_data
    
    # for artificial imbalanced setting: only the last imb_class_num classes are imbalanced
    imb_class_num = n_cls // 2 ########################

    
    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])
    min_num = np.min(class_num_list[:n_cls-imb_class_num])
    

    for i in range(n_cls):
        if imb_ratio < 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
            new_class_num_list.append(min(int(max_num*imb_ratio), class_num_list[i]))
            
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list
    

    

    if imb_ratio < 1: ### train data에서 불균형 mask
        data_train_mask, idx_info, edge_mask = split_semi_dataset(len(data.x), n_data, n_cls, class_num_list, idx_info, data.x.device, edge_index)


    
    idx_train = torch.Tensor(data_train_mask)
    idx_val = torch.Tensor(data_val_mask)
    idx_test = torch.Tensor(data_test_mask)
    
    if fake_samples == True:
        print("-------fake samples generate----------")
    
        num_fake_list =[]
        mixed_nodes = []
        for i in range(n_cls): # i = class, label
            if n_data[i] > class_num_list[i]:
                print(f"--------{i} class ----------{n_data[i] - class_num_list[i]}-------")
                num_fake_list.append(n_data[i] - class_num_list[i])
                if mix_node_init == True:
                    for _ in range(n_data[i] - class_num_list[i]):
                        minor_random = idx_info[i][torch.randperm(len(idx_info[i]))[:2]]
                        node1 = features[minor_random[0]].clone()
                        node2 = features[minor_random[1]].clone()
                        lam = torch.rand(1)
                        mixed_node = lam * node1 + (1 - lam) * node2
                        mixed_nodes.append(mixed_node.unsqueeze(0)) 
            else:
                num_fake_list.append(n_data[i] - class_num_list[i])

        if mix_node_init == True:
            print("mix node init")
            mixed_nodes = torch.cat(mixed_nodes, dim=0)
            fake_feature = mixed_nodes
        else:
            print("zero node init")
            
            fake_feature = torch.zeros((sum(num_fake_list),features.shape[1]))

    num_fake = sum(num_fake_list)
    
    tensor_list = []
    for i in range(n_cls):
        if i > n_cls-1-imb_class_num:
            tensor = torch.full((1,num_fake_list[i]), i)
            tensor_list.append(tensor) 

    fake_lables = torch.cat(tensor_list, dim=1)
    fake_idx = torch.arange(len(features), len(features)+num_fake)


    idx_train = torch.cat([idx_train, torch.zeros(len(fake_idx), dtype=torch.bool)])
    idx_val = torch.cat([idx_val, torch.zeros(len(fake_idx), dtype=torch.bool)])
    idx_test = torch.cat([idx_test, torch.zeros(len(fake_idx), dtype=torch.bool)])
    
    idx_train[-len(fake_idx):] = True
    

    features =torch.cat([features, fake_feature])
    labels =torch.cat([labels, fake_lables.squeeze(0)], dim=0)


    if not knn_size is None:
        print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
        adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    elif not epsilon is None:
        print('[ Using Epsilon-graph as input graph: {} ]'.format(epsilon))
        feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
        mask = (attention > epsilon).float()
        adj = attention * mask
        adj = (adj > 0).float()
        adj = sp.csr_matrix(adj)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    else:
        print('[ Using ground-truth input graph ]')

        if prob_del_edge is not None:
            adj = graph_delete_connections(prob_del_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)


        elif prob_add_edge is not None:
            adj = graph_add_connections(prob_add_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)

        else:
            adj = adj + sp.eye(adj.shape[0])
            adj_norm = normalize_sparse_adj(adj)
            


        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())
     
        if imb_ratio < 1:
            row, col = edge_index[:,edge_mask]
            adj_mask = torch.ones((adj_norm.shape[0],adj_norm.shape[1]), dtype=torch.bool )
    
            adj_mask[row,col] = False
            adj_norm = adj_norm*adj_mask

            if fake_samples==True:
                n = adj_norm.size(0)
                zero_col = torch.zeros(n, num_fake, device=adj_norm.device)
                adj_norm = torch.cat([adj_norm, zero_col], dim=1)
                zero_row = torch.zeros(num_fake, n + num_fake, device=adj_norm.device)
                adj_norm = torch.cat([adj_norm, zero_row], dim=0)
            else:
                fake_idx=None
    

    return adj_norm, features, labels, idx_train, idx_val, idx_test, fake_idx, new_class_num_list
        

def load_citeseer_data(args, data_dir, dataset_str, knn_size=None, epsilon=None, knn_metric='cosine', prob_del_edge=None, prob_add_edge=None, seed=1234, sparse_init_adj=False):   
    fake_samples=args['fake_samples']
    helper_ratio=args['helper_ratio']
    imb_ratio=args['imb_ratio']
    mix_node_init = args['mix_node_init']

    

                
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
                
    print("path",os.path)

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)


    # Fix citeseer dataset (there are some isolated nodes in the graph)
    # Find isolated nodes, add them as zero-vecs into the right position
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range-min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range-min(test_idx_range), :] = ty
    ty = ty_extended

    raw_features = sp.vstack((allx, tx)).tolil()
    raw_features[test_idx_reorder, :] = raw_features[test_idx_range, :]
    features = normalize_features(raw_features)
    raw_features = torch.Tensor(raw_features.todense())
    features = torch.Tensor(features.todense())

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    row, col = np.nonzero(adj) ###
    edge_index = np.vstack((row, col))###
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.argmax(labels, axis=1))
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()
    

    idx_train = sample_mask(idx_train, labels.shape[0])
    idx_val = sample_mask(idx_val, labels.shape[0])
    idx_test = sample_mask(idx_test, labels.shape[0])   

    idx_train = torch.BoolTensor(idx_train)
    idx_val = torch.BoolTensor(idx_val)
    idx_test = torch.BoolTensor(idx_test)
    

    stats = labels[idx_train]
    n_cls = labels.max().item() + 1
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(labels, n_cls, idx_train)
    
   
    
    class_num_list = n_data

    imb_class_num = n_cls // 2 
        
    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])
    min_num = np.min(class_num_list[:n_cls-imb_class_num])
 

    for i in range(n_cls):
        if imb_ratio < 1 and i > n_cls-1-imb_class_num:
            new_class_num_list.append(min(int(max_num*imb_ratio), class_num_list[i]))
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list
    
    
    if imb_ratio < 1: 
        idx_train, idx_info, edge_mask = split_semi_dataset(features.shape[0], n_data, n_cls, class_num_list, idx_info, features.device, edge_index)



    if fake_samples == True:
        print("-------fake samples generate----------")
    
        num_fake_list =[]
        mixed_nodes = []
        for i in range(n_cls): # i = class, label
            if n_data[i] > class_num_list[i]:
                print(f"--------{i} class ----------{n_data[i] - class_num_list[i]}-------")
                num_fake_list.append(n_data[i] - class_num_list[i])

                if mix_node_init == True:  #mix node
                    for _ in range(n_data[i] - class_num_list[i]):
                        minor_random = idx_info[i][torch.randperm(len(idx_info[i]))[:2]]
                        node1 = features[minor_random[0]].clone()
                        node2 = features[minor_random[1]].clone()
                        lam = torch.rand(1)
                        mixed_node = lam * node1 + (1 - lam) * node2
                        mixed_nodes.append(mixed_node.unsqueeze(0)) 
            else:
                num_fake_list.append(n_data[i] - class_num_list[i])
                

        if mix_node_init == True:
            print("mix node init")
            mixed_nodes = torch.cat(mixed_nodes, dim=0)
            fake_feature = mixed_nodes
        else:
            print("zero node init")
            fake_feature = torch.zeros((sum(num_fake_list),features.shape[1]))

        num_fake = sum(num_fake_list)
        
   
        tensor_list = []
        for i in range(n_cls):
            if i > n_cls-1-imb_class_num:
                tensor = torch.full((1,num_fake_list[i]), i)
                tensor_list.append(tensor) 
        fake_lables = torch.cat(tensor_list, dim=1)
        fake_idx = torch.arange(len(features), len(features)+num_fake)
        idx_train = torch.cat([idx_train, torch.zeros(len(fake_idx), dtype=torch.bool)])
        idx_val = torch.cat([idx_val, torch.zeros(len(fake_idx), dtype=torch.bool)])
        idx_test = torch.cat([idx_test, torch.zeros(len(fake_idx), dtype=torch.bool)])
        idx_train[-len(fake_idx):] = True
        
        features =torch.cat([features, fake_feature])
        labels =torch.cat([labels, fake_lables.squeeze(0)], dim=0)
        
    if not knn_size is None:
        print('[ Using KNN-graph as input graph: {} ]'.format(knn_size))
        adj = kneighbors_graph(features, knn_size, metric=knn_metric, include_self=True)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    elif not epsilon is None:
        print('[ Using Epsilon-graph as input graph: {} ]'.format(epsilon))
        feature_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        attention = torch.mm(feature_norm, feature_norm.transpose(-1, -2))
        mask = (attention > epsilon).float()
        adj = attention * mask
        adj = (adj > 0).float()
        adj = sp.csr_matrix(adj)
        adj_norm = normalize_sparse_adj(adj)
        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    else:
        print('[ Using ground-truth input graph ]')

        if prob_del_edge is not None:
            adj = graph_delete_connections(prob_del_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)


        elif prob_add_edge is not None:
            adj = graph_add_connections(prob_add_edge, seed, adj.toarray(), enforce_connected=False)
            adj = adj + np.eye(adj.shape[0])
            adj_norm = normalize_adj(torch.Tensor(adj))
            adj_norm = sp.csr_matrix(adj_norm)

        else:
            adj = adj + sp.eye(adj.shape[0])
            adj_norm = normalize_sparse_adj(adj)


        if sparse_init_adj:
            adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
        else:
            adj_norm = torch.Tensor(adj_norm.todense())

    if imb_ratio < 1:
        row, col = edge_index[:,edge_mask]
        adj_mask = torch.ones((adj_norm.shape[0],adj_norm.shape[1]), dtype=torch.bool )
        adj_mask[row,col] = False
        adj_norm = adj_norm*adj_mask
    
    if fake_samples==True:
        n = adj_norm.size(0)
        zero_col = torch.zeros(n, num_fake, device=adj_norm.device)
        adj_norm = torch.cat([adj_norm, zero_col], dim=1)
        zero_row = torch.zeros(num_fake, n + num_fake, device=adj_norm.device)
        adj_norm = torch.cat([adj_norm, zero_row], dim=0)
    else:
        fake_idx=None
    

    return adj_norm, features, labels, idx_train, idx_val, idx_test, fake_idx, new_class_num_list


def graph_delete_connections(prob_del, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    del_adj = np.array(adj, dtype=np.float32)
    pre_num_edges = np.sum(del_adj)

    smpl = rnd.choice([0., 1.], p=[prob_del, 1. - prob_del], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    del_adj *= smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(del_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                del_adj[k, other_node] = 1
                del_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)

    cur_num_edges = np.sum(del_adj)
    print('[ Deleted {}% edges ]'.format(100 * (pre_num_edges - cur_num_edges) / pre_num_edges))
    return del_adj


def graph_add_connections(prob_add, seed, adj, enforce_connected=False):
    rnd = np.random.RandomState(seed)

    add_adj = np.array(adj, dtype=np.float32)

    smpl = rnd.choice([0., 1.], p=[1. - prob_add, prob_add], size=adj.shape) * np.triu(np.ones_like(adj), 1)
    smpl += smpl.transpose()

    add_adj += smpl
    if enforce_connected:
        add_edges = 0
        for k, a in enumerate(add_adj):
            if not list(np.nonzero(a)[0]):
                prev_connected = list(np.nonzero(adj[k, :])[0])
                other_node = rnd.choice(prev_connected)
                add_adj[k, other_node] = 1
                add_adj[other_node, k] = 1
                add_edges += 1
        print('# ADDED EDGES: ', add_edges)
    add_adj = (add_adj > 0).astype(float)
    return add_adj

