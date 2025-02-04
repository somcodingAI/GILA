import warnings
import pickle as pkl
import sys, os
import os.path as osp

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")

###

def split_semi_dataset(total_node, n_data, n_cls, class_num_list, idx_info, device, edge_index):


    new_idx_info = []
    _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool, device=device)
    for i in range(n_cls):
        if n_data[i] > class_num_list[i]:

            cls_idx = torch.randperm(len(idx_info[i]))
            cls_idx = idx_info[i][cls_idx]  ##데이터 인덱스 무작위로 섞기 
            cls_idx = cls_idx[:class_num_list[i]] # 임벨런스 비율에 따라 앞에 n개 idx만 선택 
            new_idx_info.append(cls_idx)
        else:

            new_idx_info.append(idx_info[i])
        _train_mask[new_idx_info[i]] = True
    assert _train_mask.sum().long() == sum(class_num_list)
    assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

    print(torch.sum(_train_mask, axis=0))
    
    # 제외된 노드 찾기
    excluded_nodes = []
    for i in range(n_cls):
        result =np.setdiff1d(idx_info[i].cpu(),new_idx_info[i].cpu())
        if result.size != 0:
            excluded_nodes.append(result)
            
    excluded_nodes= np.concatenate(excluded_nodes, axis=0)
    excluded_nodes = torch.from_numpy(excluded_nodes)

    # 제외된 노드에 연결된 edge마스크 만들기
    node_mask = torch.zeros(total_node, dtype=torch.bool, device=device)
    
    # 제외 노드 인지?
    node_mask[excluded_nodes] = True

    # 엣지의 출발(row)과 도착(col) 노드 추출
    row, col = edge_index[0], edge_index[1]

    # 각 엣지의 양 끝 노드가 활성화된 상태인지 확인
    row_mask = node_mask[row]
    col_mask = node_mask[col]

    # 지워야 하는 엣지 표시
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

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
###


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
    return np.array(mask, dtype=np.bool)

def load_citation_network(dataset_str,imb_ratio,fake_samples, updaters, mix_node_init,num_minor_class, sparse=None):
    if dataset_str == 'citeseer':
        return load_citeseer_network(dataset_str,imb_ratio,fake_samples, updaters, mix_node_init,num_minor_class, sparse)
  
    else:
        if dataset_str =='cora':
            dataset = 'Cora'
        elif dataset_str =='pubmed':
            dataset = 'PubMed'
        print(f"------------{dataset_str} preprocessing--------------")
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
        dataset = get_dataset(dataset, path, split_type='public')
        data = dataset[0]
        
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)        
        features = data.x
        edge_index = data.edge_index


        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        if not sparse:
            adj = np.array(adj.todense(),dtype='float32')
        else:
            adj = sparse_mx_to_torch_sparse_tensor(adj)
        

        features = torch.Tensor(features)


######## Data statistic ##
    features = normalize_features(features)
    features = torch.FloatTensor(features)

    labels = torch.LongTensor(data.y)
    data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()
    
    
    stats = labels[data_train_mask]
    n_cls = labels.max().item() + 1
    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(labels, n_cls, data_train_mask)
    class_num_list = n_data
    

    imb_class_num = n_cls // 2 


    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])
    min_num = np.min(class_num_list[:n_cls-imb_class_num])
    

    for i in range(n_cls):
        if imb_ratio < 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
            new_class_num_list.append(min(int(max_num*imb_ratio), class_num_list[i]))
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list


    

    if imb_ratio < 1:
        data_train_mask, idx_info, edge_mask = split_semi_dataset(features.shape[0], n_data, n_cls, class_num_list, idx_info, features.device, edge_index)


    
    idx_train = torch.BoolTensor(data_train_mask)
    idx_val = torch.BoolTensor(data_val_mask)
    idx_test = torch.BoolTensor(data_test_mask)
    nfeats = features.shape[1]
    
    if fake_samples == True:
        print("-------fake samples generate----------")
        print(new_class_num_list)
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

        
     

        # 결과를 axis=1으로 쌓기
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


        if imb_ratio < 1:

            row, col = edge_index[:,edge_mask]
            adj_mask = np.ones((adj.shape[0],adj.shape[1]), dtype=bool )
            adj_mask[row,col] = False
            adj = adj*adj_mask

            if fake_samples==True:
                n = adj.shape[0]
                zero_col = np.zeros((n, num_fake), dtype=adj.dtype)
                adj = np.hstack([adj, zero_col])
                
                zero_row = np.zeros((num_fake, n + num_fake), dtype=adj.dtype)
                adj = np.vstack([adj, zero_row])
            else:
                fake_idx=None

                
    if fake_samples == True:
        return features, nfeats, labels, n_cls, idx_train, idx_val, idx_test, adj, fake_idx, new_class_num_list


def load_citeseer_network(dataset_str,imb_ratio,fake_samples, updaters, mix_node_init,num_minor_class, sparse=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    row, col = np.nonzero(adj) ###
    edge_index = np.vstack((row, col))###

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    featrues = features.todense()
    features = torch.FloatTensor(featrues)
    labels = torch.LongTensor(labels)


    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]

    stats = labels[train_mask]
    n_cls = labels.max().item() + 1



    n_data = []
    for i in range(n_cls):
        data_num = (stats == i).sum()
        n_data.append(int(data_num.item()))
    idx_info = get_idx_info(labels, n_cls, train_mask)


    class_num_list = n_data

    imb_class_num = n_cls // 2 
  

 
    new_class_num_list = []
    max_num = np.max(class_num_list[:n_cls-imb_class_num])

    print(class_num_list)

   
    for i in range(n_cls):
        if imb_ratio < 1 and i > n_cls-1-imb_class_num: #only imbalance the last classes
            new_class_num_list.append(min(int(max_num*imb_ratio), class_num_list[i]))
        else:
            new_class_num_list.append(class_num_list[i])
    class_num_list = new_class_num_list
    print(class_num_list)

    if imb_ratio < 1: 
        train_mask, idx_info, edge_mask = split_semi_dataset(features.shape[0], n_data, n_cls, class_num_list, idx_info, features.device, edge_index)

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    if fake_samples == True:
        print("-------fake samples generate----------")
        print(new_class_num_list)
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

        # 결과를 axis=1으로 쌓기
        if mix_node_init == True:
            print("mix node init")
            mixed_nodes = torch.cat(mixed_nodes, dim=0)
            fake_feature = mixed_nodes
        else:
            print("zero node init")
            
            fake_feature = torch.zeros((sum(num_fake_list),features.shape[1]))


        num_fake = sum(num_fake_list)
        
        fake_feature = torch.zeros((num_fake,features.shape[1]))
        
        tensor_list = []
        for i in range(n_cls):
            if i > n_cls-1-imb_class_num:
                tensor = torch.full((1,num_fake_list[i]), i)
                tensor_list.append(tensor) 
    
        fake_lables = torch.cat(tensor_list, dim=1)


   
        fake_idx = torch.arange(len(features), len(features)+num_fake)

        train_mask = torch.cat([train_mask, torch.zeros(len(fake_idx), dtype=torch.bool)])
        val_mask = torch.cat([val_mask, torch.zeros(len(fake_idx), dtype=torch.bool)])
        test_mask = torch.cat([test_mask, torch.zeros(len(fake_idx), dtype=torch.bool)])
        
        train_mask[-len(fake_idx):] = True
        
        features =torch.cat([features, fake_feature])
        labels =torch.cat([labels, fake_lables.squeeze(0)], dim=0)


        if imb_ratio < 1:


            row, col = edge_index[:,edge_mask]
            adj_mask = np.ones((adj.shape[0],adj.shape[1]), dtype=bool )
            adj_mask[row,col] = False
            adj = adj*adj_mask
        
  
            if fake_samples==True:
                n = adj.shape[0]
                zero_col = np.zeros((n, num_fake), dtype=adj.dtype)
                adj = np.hstack([adj, zero_col])
                
                zero_row = np.zeros((num_fake, n + num_fake), dtype=adj.dtype)
                adj = np.vstack([adj, zero_row])
            else:
                fake_idx=None

                
    if fake_samples == True:
        return features, nfeats, labels, n_cls, idx_train, idx_val, idx_test, adj, fake_idx, new_class_num_list

def load_data(args):
    return load_citation_network(args.dataset,args.imb_ratio, args.fake_samples, args.updaters, args.mix_node_init,args.num_minor_class, sparse=args.sparse)