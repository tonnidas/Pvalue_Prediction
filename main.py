# Importing all necessary python libraries 
import os
import scipy
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from model import predict_pval, special_case
from metrics import regression_metrics, regression_metrics_special_case_3
# ----------------------------------------


# ----------------------------------------
# def construct(embedding, data_name):
#     if embedding == 'Node2Vec':
#         print('Did not prepare to execute for Node2Vec yet')
#         exit(0)    # Branch closed temporarily as Node2Vec embedding performance is not good
#     else: 
#         emb_file = 'pickles/generated_embeddings/{}_{}.pickle'.format(embedding, data_name)
#         pval_file = '../graph-data/Alzheimers_Disease_Graph/Processed/{}_pval.pickle'.format(data_name)
#         with open(emb_file, 'rb') as handle: featureMatrix = pickle.load(handle)
#         with open(pval_file, 'rb') as handle: pvalueMatrix = pickle.load(handle)
#         # pvalueMatrix = pvalueMatrix.todense()
#         featureMatrix = featureMatrix.to_numpy()
#         print(featureMatrix)
#         print('pvalueMatrix.shape', pvalueMatrix.shape)
#         print(pvalueMatrix)

#     return featureMatrix, pvalueMatrix
# # ----------------------------------------

# ----------------------------------------
def construct(embedding, data_name):
    if embedding == 'Node2Vec':
        print('Did not prepare to execute for Node2Vec yet')
        exit(0)    # Branch closed temporarily as Node2Vec embedding performance is not good
    else: 
        emb_file_with_pval = 'pickles/generated_embeddings_pval_split/{}_{}_with_pval.pickle'.format(embedding, data_name)
        emb_file_without_pval = 'pickles/generated_embeddings_pval_split/{}_{}_without_pval.pickle'.format(embedding, data_name)
        with open(emb_file_with_pval, 'rb') as handle: emb_with_pval_df = pickle.load(handle)
        with open(emb_file_without_pval, 'rb') as handle: emb_without_pval_df = pickle.load(handle)
        
        # pvalueMatrix = pvalueMatrix.todense()
        predicting_pval = emb_with_pval_df['pval']
        predicting_pval = predicting_pval.to_numpy()
        emb_with_pval_df = emb_with_pval_df.drop('pval', axis=1).drop('nodeId', axis=1) # remove nodeId and pval columns
        featureMatrix = emb_with_pval_df.to_numpy()
        print('featureMatrix and predicting_pval', featureMatrix.shape, predicting_pval.shape)

        nodeId_emptyPval = emb_without_pval_df['nodeId'].values.tolist() # store nodeId for empty pval
        emb_without_pval_df = emb_without_pval_df.drop('pval', axis=1).drop('nodeId', axis=1) # remove nodeId and pval columns
        featureMatrix_of_emptyPval = emb_without_pval_df.to_numpy()
        print('featureMatrix_of_emptyPval.shape', featureMatrix_of_emptyPval.shape)
        print('nodeId_emptyPval', 'len', len(nodeId_emptyPval), 'min', min(nodeId_emptyPval), 'max', max(nodeId_emptyPval))

    return featureMatrix, predicting_pval, featureMatrix_of_emptyPval, nodeId_emptyPval
# ----------------------------------------

# ----------------------------------------
# Params:
#   dataset                 = a string containing dataset name
#   model_name              = a string
#   rand_state_for_split    = an integer
#   embedding               = a string
#   predicting_attribute    = a string

# Return values:
#   scores                  = a float
# def get_settings(dataset_attributes, dataset_edges, model, predicting_attribute, prediction_type, selected_features):
def do_category_specific_task_prediction(data_name, model_name, rand_state_for_split, embedding, predicting_speciality):   # predicting_speciality can be 1, 2, 3
    
    if predicting_speciality == 1:  # Normal prediction
        # Get features and labels
        featureMatrix, pvalueMatrix, _, _ = construct(embedding, data_name)
        print("Featurs and y collected ___________________________ ")

        # Get pval_test and pval_predicted
        y_train, predicted_train_pvals, y_test, predicted_pvals = predict_pval(featureMatrix, pvalueMatrix, model_name, rand_state_for_split)

        # Get evaluation metric values
        scores = regression_metrics(y_train, predicted_train_pvals, y_test, predicted_pvals)
        return scores
    
    elif predicting_speciality == 2:  # predict only 
        # Get features and labels
        featureMatrix, pvalueMatrix, featureMatrix_of_emptyPval, nodeId_emptyPval = construct(embedding, data_name)
        print("Featurs and y and featureMatrix_of_emptyPval collected ___________________________ ")

        # Get pval_test and pval_predicted
        y_train, predicted_train_pvals, y_test, predicted_pvals = special_case(featureMatrix, pvalueMatrix, model_name, rand_state_for_split, featureMatrix_of_emptyPval, nodeId_emptyPval, predicting_speciality)
        
        # Get evaluation metric values
        scores = regression_metrics(y_train, predicted_train_pvals, y_test, predicted_pvals)
        return scores
    
    elif predicting_speciality == 3:
        # Get features and labels
        featureMatrix, pvalueMatrix, featureMatrix_of_emptyPval, nodeId_emptyPval = construct(embedding, data_name)
        print("Featurs and y and featureMatrix_of_emptyPval collected ___________________________ ")

        # Get pval_test and pval_predicted
        y_train, predicted_train_pvals, y_test, predicted_pvals = special_case(featureMatrix, pvalueMatrix, model_name, rand_state_for_split, featureMatrix_of_emptyPval, nodeId_emptyPval, predicting_speciality)
        
        # Get evaluation metric values
        scores = regression_metrics_special_case_3(y_train, predicted_train_pvals)
        return scores

    else:
        print('Chose predicting_speciality from 1 to 3')
        exit(0)
# ----------------------------------------






# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
args = parser.parse_args()
data_name = args.dataset   # 'Alzheimer'
print('Arguments:', args)
# python main.py --dataset=Alzheimer
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
embedding = 'GCN'           # 'GCN' or 'Attri2Vec' or 'GraphSAGE' or 'Node2Vec'
model_name = 'Linear'       # 'RandomForest_randomized' or 'RandomForest_grid' or 'NeuralNet' or 'NeuralNet_hyper' or 'Linear' or 'Ridge' or 'SVM'
predicting_speciality = 2   # 1, 2, 3

# data_name = 'Alzheimer' 
# model_name = 'RandomForest_randomized' or 'RandomForest_grid' or 'NeuralNet' or 'NeuralNet_hyper'
# predicting_attribute = 'pval' or 'chromosome' or 'jaccard_similarity' 
# embedding = 'GCN' or 'Node2Vec' or 'Attri2Vec' or 'GraphSAGE'
scores = do_category_specific_task_prediction(data_name, model_name, 42, embedding, predicting_speciality) 

e_dict = dict()
metric_name = ['mse', 'rmse', 'mae']

e_dict['embedding'] = [embedding]
e_dict['model'] = [model_name]
e_dict['Train_MSE'] = [round(scores[0], 22)]
e_dict['Train_RMSE'] = [round(scores[1], 22)]
e_dict['Train_MAE'] = [round(scores[2], 22)]
e_dict['MSE'] = [round(scores[3], 22)]
e_dict['RMSE'] = [round(scores[4], 22)]
e_dict['MAE'] = [round(scores[5], 22)]

resDf = pd.DataFrame.from_dict(e_dict, orient='columns')
resDf.to_excel('Result/Regression_performance/{}_{}_case_{}.xlsx'.format(embedding, model_name, predicting_speciality)) 
print(resDf)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------