# ----------------------------------------
# Importing all necessary python libraries 
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
# ----------------------------------------


def regression_metrics(y_train, predicted_train_pvals, y_test, predicted_pvals):
    train_mse = mean_squared_error(y_train, predicted_train_pvals)
    train_rmse = train_mse**.5
    train_mae = mean_absolute_error(y_train, predicted_train_pvals)
    print('Train_MSE:', train_mse, 'Train_RMSE:', train_rmse, 'Train_MAE:', train_mae)

    mse = mean_squared_error(y_test, predicted_pvals)
    rmse = mse**.5
    mae = mean_absolute_error(y_test, predicted_pvals)
    print('MSE:', mse, 'RMSE:', rmse, 'MAE:', mae)

    scores = [train_mse, train_rmse, train_mae, mse, rmse, mae]

    # To observe how deviant the predicted values are in comparisn to actual test set values
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.scatter(y_test,predicted_pvals,color='green')
    plt.show()
    return scores

def regression_metrics_special_case_3(y_train, predicted_train_pvals):
    train_mse = mean_squared_error(y_train, predicted_train_pvals)
    train_rmse = train_mse**.5
    train_mae = mean_absolute_error(y_train, predicted_train_pvals)
    print('Train_MSE:', train_mse, 'Train_RMSE:', train_rmse, 'Train_MAE:', train_mae)

    scores = [train_mse, train_rmse, train_mae, 0, 0, 0]

    # To observe how deviant the predicted values are in comparisn to actual test set values
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    plt.scatter(y_train, predicted_train_pvals, color='red')
    plt.show()
    return scores