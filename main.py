import torch_geometric.utils.convert as cv
from torch_geometric.data import NeighborSampler as RawNeighborSampler
import pandas as pd
from utils import *
import warnings
import argparse
warnings.filterwarnings('ignore')
import networkx as nx
from sklearn.metrics import roc_auc_score
from models import *
import numpy as np
import random
import torch
from data import *
import time 

def set_seeds(n):
    seed = int(n)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed = 22
set_seeds(seed)

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run myProject.")
    parser.add_argument('--attribute_folder', nargs='?', default='dataset/attribute/')
    parser.add_argument('--data_folder', nargs='?', default='dataset/graph/')
    parser.add_argument('--alignment_folder', nargs='?', default='dataset/alignment/',
                         help="Make sure the alignment numbering start from 0")
    parser.add_argument('--k_hop', nargs='?', default=2)  
    parser.add_argument('--hid_dim', nargs='?', default=150) 
    parser.add_argument('--train_ratio', nargs='?', default= 0.1)
    parser.add_argument('--patient', nargs='?', default= 100)
    parser.add_argument('--graphname', nargs='?', default='douban') 
    parser.add_argument('--mode', nargs='?', default='not_perturbed', help="not_perturbed or perturbed") 
    parser.add_argument('--edge_portion', nargs='?', default=0.05,  help="a param for the perturbation case")  
    
    return parser.parse_args()

args = parse_args()


''' ------------------------ Run Grad-Align -----------------------------  '''

if __name__ == "__main__":
    start_time = time.time()
    G1, G2, alignment_dict, alignment_dict_reversed, idx1_dict, idx2_dict = na_dataloader(args)
    ASSISTANT = ASSISTANT(G1, G2, args.graphname, args.k_hop, args.hid_dim, alignment_dict, alignment_dict_reversed, \
                                      args.train_ratio, args.patient, idx1_dict, idx2_dict, alpha = G2.number_of_nodes() / G1.number_of_nodes(), beta = 1)    
    ASSISTANT.run_algorithm()
    end_time = time.time()
    print(f"Dataset: {args.graphname} took {(end_time-start_time):.0f} seconds")