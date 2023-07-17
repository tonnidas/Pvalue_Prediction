# Importing all necessary python libraries 
import os
import scipy
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression    # To run linear regression 
from sklearn.linear_model import Ridge               # To run ridge regression
from sklearn.svm import SVR                          # To run SVM regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer, accuracy_score


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, Activation, Flatten

from bayes_opt import BayesianOptimization

from tensorflow.keras.layers import Dropout

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# ----------------------------------------
# ----------------------------------------


# ----------------------------------------
# def randomForest_classifier_hyper(X_train, X_test, y_train, y_test):
    # # Number of trees in random forest
    # n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 100)]

    # # Number of features to consider at every split
    # max_features = ['auto', 'log2', 'sqrt']

    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)

    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]

    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]

    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]

    # # Create the random grid
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}

    # print('running random forest with hyperparameter')

    # rf = RandomForestClassifier()
    # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    # rf_random.fit(X_train, y_train)

    # # predict
    # predicted_labels_train = rf_random.predict(X_train)
    # predicted_labels_test = rf_random.predict(X_test)

    # print('randomForest with hyperparameter completed')
    # # summarize result
    # print('Best Hyperparameters: %s' % rf_random.best_params_)

    # return predicted_labels_test
# ----------------------------------------


# ----------------------------------------
def linear_regression(X_train, X_test, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def ridge_regression(X_train, X_test, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def svm_regression(X_train, X_test, y_train):
    model = SVR(kernel = 'rbf')
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def randomForest_regressor_gridSearch(X_train, X_test, y_train):

    # Number of trees in random forest
    n_estimators = [50, 100, 150, 200, 500]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [10, 20, 50, 70, 100]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    regressor = RandomForestRegressor()
    regressor_grid = GridSearchCV(estimator = regressor, param_grid = random_grid, cv = 5, verbose=2, n_jobs = -1)
    regressor_grid.fit(X_train, y_train)

    # regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    # regressor.fit(X_train, y_train)

    y_train_pred = regressor_grid.predict(X_train)
    y_pred = regressor_grid.predict(X_test)
    y_train_pred = pd.Series(y_train_pred)
    y_pred = pd.Series(y_pred)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def randomForest_regressor_randomized(X_train, X_test, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 50)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    
    regressor = RandomForestRegressor()
    regressor_randomized = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, random_state=42, n_jobs = -1)
    regressor_randomized.fit(X_train, y_train)

    y_train_pred = regressor_randomized.predict(X_train)
    y_train_pred = pd.Series(y_train_pred)
    y_pred = regressor_randomized.predict(X_test)
    y_pred = pd.Series(y_pred)
    return y_train_pred, y_pred
# ----------------------------------------


# ----------------------------------------
def neuralNet_regressor(X_train, X_test, y_train):
    NN_model = Sequential()
    NN_model.add(Dropout(0.2, input_dim = X_train.shape[1]))
    NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dropout(0.1))
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile model
    NN_model.compile(loss='mean_squared_error', optimizer=Adam(0.00001), metrics=['mean_squared_error'])
    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    NN_model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=callbacks_list)
    print(NN_model.summary())
    NN_model.load_weights(filepath) # load it

    y_train_pred = NN_model.predict(X_train)
    y_pred = NN_model.predict(X_test)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def neuralNet_regressor_hyper(X_train, X_test, y_train):

    # Create the model  .....................................................................................................
    def create_model(optimizer, activation):
        neurons = 16 
        model = Sequential()
        model.add(Dropout(0.2, input_dim = X_train.shape[1]))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(128, kernel_initializer='normal', activation=activation))
        model.add(Dense(128, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dense(neurons, kernel_initializer='normal', activation=activation))
        model.add(Dropout(0.1))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])  # Compile the model
        return model
    # Model creation done  ...................................................................................................

    param_grid = dict(epochs = [10, 20, 100], 
                  batch_size = [32, 64, 128, 512], 
                  optimizer = ['SGD', 'Adam', 'RMSprop'], 
                  activation= ['relu', 'sigmoid', 'tanh'])
    
    kr = KerasRegressor(build_fn=create_model, verbose=1)
    NN_model = GridSearchCV(estimator = kr, param_grid=param_grid, cv=3)  
    grid_result = NN_model.fit(X_train, y_train)

    # print the best parameters
    print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
    params_dict = grid_result.best_params_

    # re-run with best param
    developed_model = create_model(optimizer=params_dict['optimizer'], activation=params_dict['activation'])
    developed_model.fit(X_train, y_train, validation_split=0.2, epochs=params_dict['epochs'], batch_size=params_dict['batch_size'])
                        
    y_train_pred = developed_model.predict(X_train)
    y_pred = developed_model.predict(X_test)
    return y_train_pred, y_pred
# ----------------------------------------

# ----------------------------------------
def special_case(featureMatrix, pvalueMatrix, model_name, rand_state, featureMatrix_of_emptyPval, nodeId_emptyPval, case):
    print('======= Reached special case', case)

    X_train, X_test, y_train, y_test = train_test_split(featureMatrix, pvalueMatrix, random_state=rand_state, test_size=0.30, shuffle=True)

    if case == 2:
        print('case 2 =============')
        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted_train_pvals = model.predict(X_train)
        predicted_test_pvals = model.predict(X_test)

        # predict pvalues of nodes that do not have pvalues and store in excel file
        predicted_pvalues_of_emptyPval = model.predict(featureMatrix_of_emptyPval)
        resDf = pd.DataFrame(predicted_pvalues_of_emptyPval, columns = ['Pvals'])
        resDf['nodeId'] = nodeId_emptyPval # add the nodId column from nodeIds of empty pval
        f = 'Result/Predicted_pvalues/Predicted_pvals_from_case_2_{}_{}.xlsx'.format('GCN', 'Linear')
        resDf.to_excel(f) 
        print(resDf)

        return y_train, predicted_train_pvals, y_test, predicted_test_pvals

    elif case == 3:
        print('case 3 =============')
        model = LinearRegression()
        model.fit(featureMatrix, pvalueMatrix)
        predicted_train_pvals = model.predict(featureMatrix)

        # predict pvalues of nodes that do not have pvalues and store in excel file
        predicted_pvalues_of_emptyPval = model.predict(featureMatrix_of_emptyPval)
        resDf = pd.DataFrame(predicted_pvalues_of_emptyPval, columns = ['Pvals'])
        resDf['nodeId'] = nodeId_emptyPval # add the nodId column from nodeIds of empty pval
        print('-------------------------------------------------------------=============================')
        print(resDf, 'Tonni')
        f = 'Result/Predicted_pvalues/Predicted_pvals_from_case_3_{}_{}.xlsx'.format('GCN', 'Linear')
        resDf.to_excel(f) 
        print(resDf)
        return pvalueMatrix, predicted_train_pvals, None, None

    else:
        print('There are only case 2 and 3')
        exit(0)
# ----------------------------------------

# ----------------------------------------
def predict_pval(featureMatrix, pvalueMatrix, model, rand_state): 
    print(featureMatrix.shape, pvalueMatrix.shape)
    X_train, X_test, y_train, y_test = train_test_split(featureMatrix, pvalueMatrix, random_state=rand_state, test_size=0.30, shuffle=True)

    predicted_pvals = [0]
    if model == 'RandomForest_grid':
        predicted_train_pvals, predicted_pvals = randomForest_regressor_gridSearch(X_train, X_test, y_train)
    elif model == 'RandomForest_randomized':
        predicted_train_pvals, predicted_pvals = randomForest_regressor_randomized(X_train, X_test, y_train)
    elif model == 'NeuralNet':
        predicted_train_pvals, predicted_pvals = neuralNet_regressor(X_train, X_test, y_train)
    elif model == 'NeuralNet_hyper':
        predicted_train_pvals, predicted_pvals = neuralNet_regressor_hyper(X_train, X_test, y_train)
    elif model == 'Linear':
        predicted_train_pvals, predicted_pvals = linear_regression(X_train, X_test, y_train)
    elif model == 'Ridge':
        predicted_train_pvals, predicted_pvals = ridge_regression(X_train, X_test, y_train)
    elif model == 'SVM':
        predicted_train_pvals, predicted_pvals = svm_regression(X_train, X_test, y_train)
    else:
        print('Sorry! The model name is not right!')
        exit(0)

    return y_train, predicted_train_pvals, y_test, predicted_pvals, nodeId_train, nodeId_test
    # return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, f1_weighted, adj_RI
# ----------------------------------------