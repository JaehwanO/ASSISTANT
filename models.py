
import os.path as osp

from utils import *
from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression
import random
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv, GINConv, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from scipy.sparse import csr_matrix

from torch.nn import init

import networkx as nx
import numpy as np
import pandas as pd
import copy 

from torch_geometric.utils import *
from torch_geometric.data import NeighborSampler as RawNeighborSampler

from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from tqdm import tqdm
from tqdm import trange
import time
import IPython

class ASSISTANT:
    def __init__(self, G1, G2, graphname, k_hop, hid, alignment_dict, alignment_dict_reversed, train_ratio, patient, idx1_dict, idx2_dict, alpha,beta):

        self.G1 = G1
        self.G2 = G2
        self.layer = k_hop
        
        self.att_s = None
        self.att_t = None
        self.iter = 10
        self.epochs = 1000     
        self.hid_channel = hid
        
        self.default_weight = 1.0
        
        self.device = torch.device('cpu')    
        #self.device = torch.device('cuda')   
        self.alignment_dict = alignment_dict
        self.alignment_dict_reversed =alignment_dict_reversed

        self.train_ratio = train_ratio
        self.patient = patient
        self.idx1_dict = idx1_dict 
        self.idx2_dict = idx2_dict 

        self.gamma = 1
        self.lp_thresh = 0.7
        
        self.alpha = alpha
        self.beta = beta
        
        self.eval_mode = True #
        self.cea_mode = False
        self.fast_mode = True
        
        self.lamb = 0.3
        
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # disable temporarily because of error
        
    def run_algorithm(self): #anchor is not considered yet
        iteration = 0

        seed_list1 = list(np.random.choice(list(self.alignment_dict.keys()), int(self.train_ratio * len(self.alignment_dict)), replace = False))
        seed_list2 = [self.alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
        self.pre_seed_list1 = seed_list1
        self.pre_seed_list2 = seed_list2
        self.G1, self.G2 = seed_link(seed_list1, seed_list2, self.G1, self.G2)
        self.H = self.calculateH(self.gamma)

        #* 1. RWR
        rwr_score_s = rwr_scores(self.G1, seed_list1)
        rwr_score_t = rwr_scores(self.G2, seed_list2)
       
        #* 2. AdaSim
        adasim_s = compute_AdaSim(self.G1, seed_list1, self.idx1_dict, decay_factor=0.7, iterations=1, alpha_val=0.7, link_type='none')
        adasim_t = compute_AdaSim(self.G2, seed_list2, self.idx2_dict, decay_factor=0.7, iterations=1, alpha_val=0.7, link_type='none')
        
        #* 3. SimRank
        simrank_s = my_simrank(self.G1, seed_list1)
        simrank_t = my_simrank(self.G2, seed_list2)
        
        #* 4. Jaccard 
        jaccard_s = my_jaccard(self.G1, seed_list1)
        jaccard_t = my_jaccard(self.G2, seed_list2)
       
        #* 5. Degree
        # degree_s = build_degrees(self.G1, 200)
        # degree_t = build_degrees(self.G2, 200)

        final_att_s = np.concatenate((rwr_score_s,adasim_s,simrank_s,jaccard_s), axis=1)
        final_att_t = np.concatenate((rwr_score_t,adasim_t,simrank_t,jaccard_t), axis=1)
        # final_att_t_ = np.random.permutation(final_att_t) # for ablation
        self.att_s = final_att_s
        self.att_t = final_att_t
        
        model = myGIN(len(self.att_s.T), hidden_channels=self.hid_channel, num_layers = self.layer)
        model = model.to(self.device)
        
        nx.set_edge_attributes(self.G1, values = self.default_weight, name = 'weight')
        nx.set_edge_attributes(self.G2, values = self.default_weight, name = 'weight')
        
        index = sorted(list(self.G1.nodes()))
        columns = sorted(list(self.G2.nodes()))
    
        start = time.time()
        
        # Start iteration

        while True:
                
            self.attr_norm_s, self.attr_norm_t =self.normalized_attribute(self.G1, self.G2)
            
            index = list(set(index) - set(seed_list1))
            columns = list(set(columns) - set(seed_list2))
             
            if self.fast_mode == True:        
                index = list(set.union(*[set(self.G1.neighbors(node)) for node in seed_list1])- set(seed_list1))
                columns = list(set.union(*[set(self.G2.neighbors(node)) for node in seed_list2])- set(seed_list2))
                
            seed_n_id_list = seed_list1 + seed_list2
            if len(columns) == 0 or len(index) == 0:
                break
            if len(self.alignment_dict) == len(seed_list1):
                break
            print('\n ------ The current iteration : {} ------'.format(iteration))
            # GNN Embedding Update
            data_s, x_s, edge_index_s, edge_weight_s = self.convert2torch_data(self.G1, self.attr_norm_s) 
            data_t, x_t, edge_index_t, edge_weight_t = self.convert2torch_data(self.G2, self.attr_norm_t)

            if iteration == 0:
                embedding1, embedding2 = self.embedding(seed_list1, seed_list2, iteration, self.epochs, x_s, edge_index_s, edge_weight_s, x_t, edge_index_t, edge_weight_t, model, data_s, data_t)
                    
            # Update graph
            print('\n start adding a seed nodes')
            seed_list1, seed_list2, anchor, S, adj2 = self.AddSeeds(embedding1, embedding2, index, columns, seed_list1, seed_list2, iteration)
            embedding_fin1 = embedding1[self.layer].detach().numpy()
            embedding_fin2 = embedding2[self.layer].detach().numpy()
            if self.cea_mode == True:                
                print('\n Edge augmentation...')
                self.G1, self.G2 = self.EvolveGraph(seed_list1, seed_list2, S)
            iteration += 1
            
        # Evaluate Performance
        print("total time : {}sec".format(int(time.time() - start)))
        print('\n Start evaluation...')
        self.Evaluation(seed_list1, seed_list2)
        
        S_prime, result = self.FinalEvaluation(S, embedding1, embedding2, seed_list1, seed_list2, self.idx1_dict, self.idx2_dict, adj2)

    
    def convert2torch_data(self, G, att):      
        
        data = from_networkx(G)
        att = torch.from_numpy(att)
        data.x = att
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        data.edge_attr = data['weight']
        edge_weight = data.edge_attr
        x = x.float()
        edge_weight = edge_weight.float()

        
        return data, x, edge_index, edge_weight
    
    def normalized_attribute(self,G1,G2):
        
        self.degarr_s = normalized_adj(G1)
        self.degarr_t = normalized_adj(G2)
        
        attr1_norm = self.att_s
        attr2_norm = self.att_t 
        
        return attr1_norm, attr2_norm
        
    def embedding(self, seed_list1, seed_list2, match_iter, epoch, x_s, edge_index_s, edge_weight_s, x_t, edge_index_t, edge_weight_t, model, data_s, data_t):

        seed_1_idx_list = [self.idx1_dict[a] for a in seed_list1]
        seed_1_idx_list = torch.LongTensor(seed_1_idx_list)        
        seed_2_idx_list = [self.idx2_dict[b] for b in seed_list2]        
        seed_2_idx_list = torch.LongTensor(seed_2_idx_list)
        
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)    
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)    
        
        A_s = nx.adjacency_matrix(self.G1)
        A_t = nx.adjacency_matrix(self.G2)
        A_hat_s_list = self.distinctive_loss(A_s.todense())
        A_hat_t_list = self.distinctive_loss(A_t.todense())
        
        t = trange(epoch, desc='EMB')            
        model.train()
        
        early_stop = 0
        min_loss = 10000000
        max_val = 0
        for ep in t:      
            
            total_loss = 0
            
            embedding_s = model.full_forward(x_s, edge_index_s)
            embedding_t = model.full_forward(x_t, edge_index_t)
            
            optimizer.zero_grad()
            loss = 0
            mapping_loss = nn.MSELoss()
            
            for i, (emb_s, emb_t, A_hat_s, A_hat_t) in enumerate(zip(embedding_s, embedding_t, A_hat_s_list, A_hat_t_list)):
                #multi-layer-loss
                if i == 0:
                    continue                
                consistency_loss_s = self.linkpred_loss(emb_s, A_hat_s)
                consistency_loss_t = self.linkpred_loss(emb_t, A_hat_t)
                loss += consistency_loss_s + consistency_loss_t
   
            loss.backward()
            optimizer.step()                
            total_loss += float(loss)
            t.set_description('EMB (total_loss=%g)' % (total_loss))
            
            S_fin = np.zeros((self.G1.number_of_nodes(),self.G2.number_of_nodes()))
            for i, (emb1, emb2) in enumerate(zip(embedding_s, embedding_t)):            
                S = torch.matmul(F.normalize(emb1), F.normalize(emb2).t())
                S = S.detach().numpy()
                
                S_fin += (1/(self.layer+1)) * S     
                
            top1_eval = self.FinalEvaluation_mid(S_fin,self.idx1_dict, self.idx2_dict, self.alignment_dict)
            
            if top1_eval > max_val:
                max_val = top1_eval
                early_stop = 0
                best_model = copy.deepcopy(model.state_dict())
            elif early_stop > self.patient:
                break
            else:
                early_stop += 1
                
        # model.load_state_dict(best_model)
        embedding_s = model.full_forward(x_s, edge_index_s)        
        embedding_t = model.full_forward(x_t, edge_index_t)
              
        return embedding_s, embedding_t

    def FinalEvaluation_mid(self, S, idx1_dict, idx2_dict, test_dict):
   
        gt_dict = test_dict
        top_1 = self.top_k_mid(S,1)

        top1_eval = compute_precision_k(top_1, gt_dict, idx1_dict, idx2_dict)
 
        return top1_eval
    
    def top_k_mid(self, S, k):
        """
        S: scores, numpy array of shape (M, N) where M is the number of source nodes,
            N is the number of target nodes
        k: number of predicted elements to return
        """
        top = np.argsort(-S)[:,:k]
        result = np.zeros(S.shape)
        for idx, target_elms in enumerate(top):
            # print(idx,target_elms)
            for elm in target_elms:
                result[idx,elm] = 1

        return result
    
    def calculateH(self, gamma):
        self.H = np.zeros((self.G1.number_of_nodes(),self.G2.number_of_nodes()))
        for i, j in zip(self.pre_seed_list1,self.pre_seed_list2):
            self.H[self.idx1_dict[i],self.idx2_dict[j]] = gamma            
        return self.H

    def AddSeeds(self, embedding1, embedding2, index, columns, seed_list1, seed_list2, iteration):

        S_fin = np.zeros((self.G1.number_of_nodes(),self.G2.number_of_nodes()))
        
        for i, (emb1, emb2) in enumerate(zip(embedding1, embedding2)):            
            S = torch.matmul(F.normalize(emb1), F.normalize(emb2).t())
            S = S.detach().numpy()
            
            S_fin += (1/(self.layer+1)) * S     
            
        try:
            S_fin = S_fin + self.H
        except:
            print("no prior anchors")
            pass
        
        sim_matrix = np.zeros((len(index) * len(columns), 3))
        
        start_tve = time.time()
        
        for i in range(len(index)):
            for j in range(len(columns)):
                sim_matrix[i * len(columns) + j, 0] = index[i] 
                sim_matrix[i * len(columns) + j, 1] = columns[j]
                sim_matrix[i * len(columns) + j, 2] = S_fin[self.idx1_dict[index[i]],self.idx2_dict[columns[j]]] 
        if len(seed_list1) != 0:
            print("Tversky sim calculation..")
            sim_matrix2 = calculate_Tversky_coefficient(self.G1, self.G2, seed_list1, seed_list2, index, columns, self.alpha, self.beta)
            sim_matrix[:, 2] *= sim_matrix2[:, 2]
        else:
            sim_matrix2 = 1 # no effecta
        sim_matrix = sim_matrix[np.argsort(-sim_matrix[:, 2])]

        print("Tversky time : {}sec".format(int(time.time() - start_tve)))
        interval_1 = time.time()
        
        seed1, seed2 = [], []
        len_sim_matrix = len(sim_matrix)
        if len_sim_matrix != 0:
            T = align_func(version='const', a=int(len(self.alignment_dict) / self.iter), b=0, i=iteration)
            nodes1, nodes2, sims = sim_matrix[:, 0].astype(int), sim_matrix[:, 1].astype(int), sim_matrix[:, 2]
            idx = np.argsort(-sims)
            nodes1, nodes2, sims = nodes1[idx], nodes2[idx], sims[idx]
            while len(nodes1) > 0 and T > 0:
                T -= 1
                node1, node2 = nodes1[0], nodes2[0]
                seed1.append(node1)
                seed2.append(node2)
                mask = np.logical_and(nodes1 != node1, nodes2 != node2)
                nodes1, nodes2, sims = nodes1[mask], nodes2[mask], sims[mask]
            sim_matrix = np.column_stack((nodes1, nodes2, sims))
        anchor = len(seed_list1)
        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)))

        print("interval_2 time : {}sec".format(int(time.time() -interval_1 )))
        
        self.Evaluation(seed_list1, seed_list2)
        
        return seed_list1, seed_list2, anchor, S_fin, sim_matrix2


    
    def EvolveGraph(self, seed_list1, seed_list2, S):

        pred1, pred2 = self.cross_link(self.lp_thresh, self.G1, self.G2, seed_list1, seed_list2, S)
        
        print("{} edges are added in total".format(len(pred1)+len(pred2)))
        self.G1.add_edges_from(pred1, weight = self.default_weight)
        self.G2.add_edges_from(pred2, weight = self.default_weight)

        return self.G1, self.G2
        
    def Evaluation(self, seed_list1, seed_list2):
        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))
        print('Prediction accuracy  at this iteration : %.2f%%'%(100 * (count-train_len) / (len(seed_list1)-train_len)))        
        print('All accuracy : %.2f%%'%(100*(count / len(self.alignment_dict))))    
        print('All prediction accuracy : %.2f%%'%(100*((count - train_len) /(len(self.alignment_dict)-train_len))))    
        
    def FinalEvaluation(self, S, embedding1, embedding2, seed_list1, seed_list2, idx1_dict, idx2_dict, adj2):
        
        count = 0

        for i in range(len(seed_list1)):
            try:
                if self.alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        train_len = int(self.train_ratio * len(self.alignment_dict))      
        print('All accuracy : %.2f%%'%(100*(count / len(self.alignment_dict)))) 
        acc = count / len(self.alignment_dict)
        
        #input embeddings are final embedding
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        
        
        if self.eval_mode == True:
            adj2 = calculate_Tversky_coefficient_final(self.G1, self.G2, seed_list1, seed_list2, index, columns, alpha = self.alpha, beta = self.beta)
            S_prime = self.adj2S(adj2, self.G1.number_of_nodes(), self.G2.number_of_nodes())
            S *= S_prime
    
        gt_dict = self.alignment_dict
        
        top_1 = top_k(S,1)
        top_5 = top_k(S,5)
        top_10 = top_k(S,10)
        
        top1_eval = compute_precision_k(top_1, gt_dict, idx1_dict, idx2_dict)
        top5_eval = compute_precision_k(top_5, gt_dict, idx1_dict, idx2_dict)
        top10_eval = compute_precision_k(top_10, gt_dict, idx1_dict, idx2_dict)
        
        print('{:.4f}'.format(top1_eval))
        print('{:.4f}'.format(top5_eval))
        print('{:.4f}'.format(top10_eval))
        
        result = '@1:' + str(round(top1_eval,4)) + ',  @5:' + str(round(top5_eval,4))+ ',  @10:' + str(round(top10_eval,4))+ ',  Acc:'+ str(round(acc,4))
        
        return S, result
        
    def adj2S(self, adj, m, n):
        # m = # of nodes in G_s
        S = np.zeros((m,n))
        index = list(self.G1.nodes())
        columns = list(self.G2.nodes())
        for i in range(m):
            for j in range(n):
                S[self.idx1_dict[index[i]], self.idx2_dict[columns[j]]] = adj[i * n + j, 2]
        return S
    
    def linkpred_loss(self, embedding, A):
        
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        
        return linkpred_losss
    
    def edge_weight_update(self, G, seed_list):
    
        for seed in seed_list:
            for nbr in list(nx.neighbors(G, seed)):
                G.edges[seed,nbr]['weight'] *= self.p 
                
    def distinctive_loss(self, A):
    
        A_hat_list = []
        A_hat_list.append(None) #empty element for future iteration
        for i in range(len(A)):
            A[i, i] = 1
        A = torch.FloatTensor(A)        
        A_cand = A
        
        for l in range(self.layer):
            
            D_ = torch.diag(torch.sum(A, 0)**(-0.5))         
            A_hat = torch.matmul(torch.matmul(D_,A),D_)
            A_hat = A_hat.float()
            A_hat_list.append(A_hat)    
            A_cand = torch.matmul(A,A_cand)
            A = A + A_cand
            
        return A_hat_list
    
    def cross_link(self, thresh, G1, G2, seed_list1, seed_list2, S):
    
        pred1 = []
        pred2 = []
    
        for i in range(len(seed_list1)):
            for j in range(i+1, len(seed_list1)):
                if (S[self.idx1_dict[seed_list1[i]],self.idx2_dict[seed_list2[i]]] > thresh 
                        and S[self.idx1_dict[seed_list1[j]],self.idx2_dict[seed_list2[j]]] > thresh):
                    
                    if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                        
                            pred1.append([min(seed_list1[i], seed_list1[j]), max(seed_list1[i], seed_list1[j])])
                        
                    if not G2.has_edge(seed_list2[i], seed_list2[j]) and G1.has_edge(seed_list1[i], seed_list1[j]):
                        pred2.append([min(seed_list2[i], seed_list2[j]), max(seed_list2[i], seed_list2[j])])
        
        return pred1, pred2


class myGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):        
        super(myGIN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Gate
        self.p = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            # nn.Tanh()
            )

        self.q = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh()
            )

        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(in_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ), train_eps=False, aggr = 'max'))
            
        init_weight(self.modules())

    def full_forward(self, x, edge_index):
        
        x_weight = self.p(x)
        x_weight = torch.sigmoid(x_weight)
        x = x.mul(x_weight)
            
        emb_list = []
        emb_list.append(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.tanh()
            emb_list.append(x)
            
        return emb_list
    
def init_weight(modules):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    
    for m in modules:
        if isinstance(m, nn.Linear):
            m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
    
def seed_link(seed_list1, seed_list2, G1, G2):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return G1, G2

def normalized_adj(G):
    # make sure ordering has ascending order
    deg = dict(G.degree)
    deg = sorted(deg.items())
    deglist = [math.pow(b, -0.5) for (a,b) in deg]
    degarr = np.array(deglist)
    degarr = np.expand_dims(degarr, axis = 0)
    return degarr.T

def align_func(version, a, b, i):
    
    if version == "lin":
        return int(a*i + b)
    elif version == "exp":
        return int(a**i + b)
    elif version == "log":    
        return int(math.log(a*i+b) + b)
    elif version == "const":
        return a

def calculate_Tversky(setA, setB, alpha, beta):
    setA = set(setA)
    setB = set(setB)   
    ep = 0.01
        
    inter = len(setA & setB) + ep
    diffA = len(setA - setB) 
    diffB = len(setB - setA)
     
    Tver = inter / (inter + alpha*diffA + beta*diffB)
    
    return Tver

def calculate_Tversky_coefficient(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta, alignment_dict = None):
    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict_reversed = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}
    for G1_node in index:
        for G2_node in columns:
            Tversky_dict[(G1_node, G2_node)] = Tversky_dict.get((G1_node, G2_node), 0) + calculate_Tversky(adj.neighbors(G1_node), adj.neighbors(G2_node + shift + 1), alpha, beta)
    
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)
    return sim_matrix

def calculate_Tversky_coefficient_final(G1, G2, seed_list1, seed_list2, index, columns, alpha, beta):
    shift = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (shift + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (shift + 1)
        seed2_dict[i + 2 * (shift + 1)] = seed_list2[i] + shift + 1
        seed2_dict_reversed[seed_list2[i] + shift + 1] = i + 2 * (shift + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += shift + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x:to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x:to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    Tversky_dict = {}  
    for G1_node in index:
        for G2_node in columns:
            Tversky_dict[G1_node, G2_node] = 0
            g1 = to_seed(G1_node, seed1_dict_reversed)
            g2 = to_seed(G2_node + shift + 1, seed2_dict_reversed)
            Tversky_dict[G1_node, G2_node] += calculate_Tversky(adj.neighbors(g1), adj.neighbors(g2), alpha, beta)
    Tversky_dict = [[x[0][0], x[0][1], x[1]] for x in Tversky_dict.items()]
    sim_matrix = np.array(Tversky_dict)
    return sim_matrix

def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:,:k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
        
    return result

def compute_precision_k(top_k_matrix, gt, idx1_dict, idx2_dict):
    n_matched = 0

    for key, value in gt.items():
        if top_k_matrix[idx1_dict[key], idx2_dict[value]] == 1:
            n_matched += 1
    return n_matched/len(gt)
    
def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)
    return 1 / (1 + np.exp(-input))

def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))


def build_degrees(G, num):
    neighbor_degrees = defaultdict(list)
    degree_feat= []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        for neigh in neighbors:
            neighbor_degrees[node].append(G.degree(neigh))
        random_degree = np.log(np.random.choice(neighbor_degrees[node], num, replace= True))
        degree_feat.append(random_degree)
    degree_feat_fin = np.array(degree_feat)

    return degree_feat_fin

def rwr_scores(G, anchors):
    n = G.number_of_nodes()
    score = []
    for i, anchor in enumerate(anchors):
        s = nx.pagerank_scipy(G, personalization={anchor: 1})
        s_list = [0] * n

        for i, node in enumerate (list(G.nodes())):
            s_list[i] = s[node]

        score.append(s_list)
    rwr_score = np.array(score).T
    return rwr_score 

def compute_AdaSim (graph, selected_anchor_node, id2idx, decay_factor, iterations, alpha_val, link_type='none'):

    #============================================================================================
        # reading graph and computing 'weights' matrix
    #============================================================================================    
    G = graph
    nodes = list(G.nodes())        # sorted list of all nodes
    adj = nx.adjacency_matrix(G, nodelist=nodes, weight=None)      # V*V adjacency matrix
    ## applied to both out-link for directed graphs or undirecetd graphs
    degrees = adj.sum(axis=1).T   # 1*V matrix (a row vector of size V)        
    weights = csr_matrix(1/np.log(degrees+math.e))  # keep weights of nodes; 1*V matrix;
    weight_matrix = csr_matrix(adj.multiply(weights)) # V*V matrix; row i have the weight of i's out-links

    #============================================================================================
        # similarity computation
    #============================================================================================        

    adamic_scores = weight_matrix * adj.T
    adamic_scores.setdiag(0)
    adamic_scores = adamic_scores/np.max(adamic_scores)  # min-max normalization
    result_matrix = decay_factor*alpha_val*adamic_scores        
    result_matrix.setdiag(1) ## Set diagonal values to one for writing results; since they are NOT used in computing similarity scores, we can skip this line for better efficiency 

    # weight_matrix = F.normalize(weight_matrix, p=1, dim=0) # row normalized weight_matrix    
    for itr in range (2, iterations+1):                            
        result_matrix.setdiag(0) ## diagonal values are set back to zero; corresponding to the ∧ opertaor in Equation (15)
        result_matrix =  decay_factor * (alpha_val* adamic_scores + (1-alpha_val) * (weight_matrix * result_matrix * weight_matrix.T)) #+ iden_matrix
        result_matrix.setdiag(1) ## setting back diagonal values to one for writing results, we can skip this line for better efficiency 
    
    AdaSim_array = np.array([[result_matrix[(id2idx[u],id2idx[v])] for v in selected_anchor_node] for u in G])
    return AdaSim_array

def my_simrank(G,selected_anchor_node):
    nodes = list(G.nodes())
    simrank_sim = nx.simrank_similarity(G)
    sim_array = np.array([[simrank_sim[u][v] for v in selected_anchor_node] for u in G])
    return sim_array
    
def my_jaccard(G, selected_anchor_node):
    sim_array = np.array([[jaccard(G,u,v) for v in selected_anchor_node] for u in G])
    return sim_array

def jaccard(G,u,v):
    union_size = len(set(G[u]) | set(G[v]))
    if union_size == 0:
        return 0
    return len(list(nx.common_neighbors(G, u, v))) / union_size