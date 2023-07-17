# Importing all necessary python libraries 
import os
import scipy
import pickle
import numpy as np
import scipy.sparse 
import pandas as pd
import networkx as nx
import stellargraph as sg
from scipy.sparse import csr_matrix
from stellargraph import StellarGraph, datasets
# from math import isclose
# from sklearn.decomposition import PCA

from embedding_4_models import run


# ================================================================================================================================================================
# Make the graph from the features and adj
def get_sg_graph(adj, features):
    print('adj shape:', adj.shape, 'feature shape:', features.shape)
    nxGraph = nx.from_scipy_sparse_array(adj)                           # make nx graph from scipy matrix

    # add features to nx graph
    for node_id, node_data in nxGraph.nodes(data=True):
        node_feature = features[node_id].todense()
        node_data["feature"] = np.squeeze(np.asarray(node_feature)) # convert to 1D matrix to array

    # make StellarGraph from nx graph
    sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="gene", edge_type_default="connects to", node_features="feature")
    print(sgGraph.info())

    return sgGraph
# ================================================================================================================================================================



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder')
parser.add_argument('--dataset')
parser.add_argument('--embedding')
args = parser.parse_args()
print('Arguments:', args)
folder_name = args.folder                                             # 'Alzheimers_Disease_Graph'
data_name = args.dataset                                              # 'Alzheimer'
embedding = args.embedding                                            # 'Attri2Vec' or 'GraphSAGE' or 'GCN' or 'Node2Vec'
print('Generate ' + embedding + ' embedding for dataset = ' + data_name + ' from folder ' + folder_name + '.')
# python embedding_generator.py --folder=Alzheimers_Disease_Graph --dataset=Alzheimer --embedding=GCN
# python embedding_generator.py --folder=Alzheimers_Disease_Graph --dataset=Alzheimer --embedding=Attri2Vec
# python embedding_generator.py --folder=Alzheimers_Disease_Graph --dataset=Alzheimer --embedding=GraphSAGE
# python embedding_generator.py --folder=Alzheimers_Disease_Graph --dataset=Alzheimer --embedding=Node2Vec
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------




# ================================================================================================================================================================
# read adj and features from pickle and prepare sg graph
featurePickleFile = '../graph-data/{}/Processed/{}_features.pickle'.format(folder_name, data_name)
adjPickleFile = '../graph-data/{}/Processed/{}_adj.pickle'.format(folder_name, data_name)
pvalPickleFile = '../graph-data/{}/Processed/{}_pval.pickle'.format(folder_name, data_name)
with open(featurePickleFile, 'rb') as handle: features = pickle.load(handle) 
with open(adjPickleFile, 'rb') as handle: adj = pickle.load(handle) 
with open(pvalPickleFile, 'rb') as handle: pval = pickle.load(handle) 

print('features.shape', features.shape)
print('adj.shape', adj.shape)
# print('len nodeList', len(nodeIdList))

# make StellarGraph and list of nodes
sgGraph = get_sg_graph(adj, features)        # make the graph
nodes_list = list(range(0, features.shape[0]))

# make StellarGraph from nx graph
# sgGraph = StellarGraph.from_networkx(nxGraph, node_type_default="people", edge_type_default="friendship", node_features="feature")
outputDf = run(embedding, data_name, nodes_list, data_name, sgGraph, 42)

outputFileName = "Result/Embedding_scores/{}_{}_roc_auc_all_data.txt".format(embedding, data_name)
f1 = open(outputFileName, "w")
f1.write("For data_name: {}, split: {}\n".format(data_name, 42))
f1.write(outputDf.to_string())
f1.close()
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------