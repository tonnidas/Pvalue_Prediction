# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import scipy
import scipy.sparse 
from scipy.sparse import csr_matrix
import pickle
import stellargraph as sg
import os
from stellargraph import StellarGraph, datasets
from math import isclose
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--dataset')
parser.add_argument('--embedding')

args = parser.parse_args()
print('Arguments:', args)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

data_name = args.dataset   # 'Alzheimer'
embedding = args.embedding # 'Attri2Vec' or 'GraphSAGE' or 'GCN'
# python pval_split.py --dataset=Alzheimer --embedding=GCN


# read adj and features from pickle and prepare sg graph
embPickleFile = 'pickles/generated_embeddings/{}_{}_all.pickle'.format(embedding, data_name)
pvalPickleFile = '../graph-data/{}/Processed/{}_pval.pickle'.format('Alzheimers_Disease_Graph', data_name, str(0))
with open(embPickleFile, 'rb') as handle: emb = pickle.load(handle) 
with open(pvalPickleFile, 'rb') as handle: pval = pickle.load(handle) 

# reset index to join with emb using sequential index
pval = pval.astype('float64').reset_index(drop=True)
all_data = emb.join(pval)

# add nodeId column
folder_name = 'Alzheimers_Disease_Graph'
nodIdListPickleFile = '../graph-data/{}/Processed/{}_node_id_list.pickle'.format(folder_name, data_name)
with open(nodIdListPickleFile, 'rb') as handle: nodeIdList = pickle.load(handle) 

print(len(nodeIdList))

all_data['nodeId'] = nodeIdList
print(all_data)

# seperate data according to non-empty and empty pval
data_pval = all_data[all_data['pval'].notna()]
print('non-empty pval:', data_pval.shape)
data_pval_file = 'pickles/generated_embeddings_pval_split/{}_{}_with_pval.pickle'.format(embedding, data_name)
with open(data_pval_file, 'wb') as handle: pickle.dump(data_pval, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_emptyPval = all_data[all_data['pval'].isnull()]
print('empty pval:', data_emptyPval.shape)
data_emptyPval_file = 'pickles/generated_embeddings_pval_split/{}_{}_without_pval.pickle'.format(embedding, data_name)
with open(data_emptyPval_file, 'wb') as handle: pickle.dump(data_emptyPval, handle, protocol=pickle.HIGHEST_PROTOCOL)