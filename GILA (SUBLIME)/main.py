import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_data
from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
import dgl
from sklearn.metrics import balanced_accuracy_score, f1_score

import random


EOS = 1e-10

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()


    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)


    def loss_cls(self, model, mask, features, labels, zero_matrix):
        logits = model(features, zero_matrix)
        logp = F.log_softmax(logits, 1)
        # print(mask.dtype, logp.dtype,labels.dtype)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu, logits


    def loss_gcl(self, model, graph_learner, features, anchor_adj, zero_matrix):

        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features, zero_matrix)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        z2, _ = model(features_v2, learned_adj, 'learner')


        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            # random.shuffle(node_idxs)
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj


    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, fake_idx, args):

        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        if args.updaters ==True:
            zero_matrix = torch.zeros((features.shape[0]-fake_idx.shape[0],features.shape[1]),dtype=torch.float).to(features.device)
        else:
            zero_matrix=None

        if torch.cuda.is_available():
            model = model.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()
            features = features.cuda()
            labels = labels.cuda()

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu, logits = self.loss_cls(model, train_mask, features, labels, zero_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu, logits = self.loss_cls(model, val_mask, features, labels, zero_matrix)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break
        best_model.eval()
        test_loss, test_accu, logits = self.loss_cls(best_model, test_mask, features, labels, zero_matrix)
        return best_val, test_accu, best_model, logits


    def train(self, args):

        torch.cuda.set_device(args.gpu)

        if args.gsl_mode == 'structure_refinement':

            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original, fake_idx, new_class_num_list = load_data(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data(args)

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []


        for trial in range(args.ntrials):

            self.setup_seed(trial)

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0])
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original)


            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)
            if args.updaters ==True:
                zero_matrix = torch.zeros((features.shape[0]-fake_idx.shape[0],features.shape[1]),dtype=torch.float).to(features.device)
            else:
                zero_matrix=None

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

 
    
            if args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner, args.updaters ,fake_idx, features)
            

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)


            if torch.cuda.is_available():
                model = model.cuda()
                graph_learner = graph_learner.cuda()
                train_mask = train_mask.cuda()
                val_mask = val_mask.cuda()
                test_mask = test_mask.cuda()
                features = features.cuda()
                labels = labels.cuda()
                if not args.sparse:
                    anchor_adj = anchor_adj.cuda()



            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_epoch = 0
                best_f1, best_bacc = 0, 0 ###

            for epoch in range(1, args.epochs + 1):

                model.train()
                graph_learner.train()

                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, zero_matrix)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)
                
                if epoch % 100 == 0:
                    print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()))

                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        model.eval()
                        graph_learner.eval()
                        f_adj = Adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, test_accu, _, logits = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels,
                                                                               nclasses, train_mask, val_mask, test_mask, fake_idx, args)
 
                        logits = torch.argmax(logits, axis=1)
                  
                        test_bacc = balanced_accuracy_score(labels[test_mask].detach().cpu(), logits[test_mask].detach().cpu())
                        test_f1 = f1_score(labels[test_mask].detach().cpu(), logits[test_mask].detach().cpu(), average='macro')      
             
                        if val_accu > best_val:
                            best_val = val_accu
                            best_val_test = test_accu
                            best_epoch = epoch
                            best_f1 = test_f1 
                            best_bacc = test_bacc 

                            print(f"updata epoch : {best_epoch}, Best test ACC:  {best_val_test.item()}, test bACC:  {best_bacc}, test f1:  {best_f1}")

                    elif args.downstream_task == 'clustering':
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        embedding = embedding.cpu().detach().numpy()

                        acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                        for clu_trial in range(n_clu_trials):
                            kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                            predict_labels = kmeans.predict(embedding)
                            cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                            acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                            acc_mr.append(acc_)
                            nmi_mr.append(nmi_)
                            f1_mr.append(f1_)
                            ari_mr.append(ari_)

                        acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)

            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                print("Trial: ", trial + 1)

                print(f"Best epoch: {best_epoch}, Best val ACC:  {best_val.item()}, Best test ACC:  {best_val_test.item()}")
                print(f"test bACC:  {best_bacc}, test f1:  {best_f1}")

      
            elif args.downstream_task == 'clustering':
                print("Final ACC: ", acc)
                print("Final NMI: ", nmi)
                print("Final F-score: ", f1)
                print("Final ARI: ", ari)

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies)


    def print_results(self, validation_accu, test_accu):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        print(s_val)
        print(s_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument('-ntrials', type=int, default=1) # 5
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_refinement",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=50)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=1)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=4000) # 2000
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=256)
    parser.add_argument('-proj_dim', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0) #
    parser.add_argument('-nlayers', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.7) #0.7
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.6) #0.6
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='att')
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200) #200
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.75)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=0.9999)
    parser.add_argument('-c', type=int, default=0)

    parser.add_argument('-imb_ratio', type=float, default=1)
    parser.add_argument('-fake_samples', default=False, type=bool) 
    parser.add_argument('-updaters', default=False, type=bool)
    parser.add_argument('-mix_node_init', default=False, type=bool)  
    parser.add_argument('-num_minor_class',default=1, type=int) 


    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args)
