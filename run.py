# -*- coding: utf-8 -*-

# Useful starting lines
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import myFunctions2 as my
%load_ext autoreload
%autoreload 2

## Load train data
####################
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, X, ids = my.load_csv_data(DATA_TRAIN_PATH)

## Transform features
#####################

def standardize_badFeatures(X):
    
    # Function that calculate the mean and std of bad features without elements equal to -999
    # Then, it remplaces -999 values by zeros, zeros won't influence the train of the model... 
    mean_x = np.zeros((X.shape[1],))
    std_x = np.zeros((X.shape[1],))
    for d in range(X.shape[1]):
        idx = np.where(X[:,d] == -999)[0]
        mean_x[d] = np.mean(np.delete(X[:,d], (idx)))
        std_x[d] = np.std(np.delete(X[:,d], (idx)))
        X[:,d] = (X[:,d]-mean_x[d])/std_x[d]
        X[idx,d] = 0
    return X, mean_x, std_x


def clean_data(X):

    # find indices of features that have at least one value -999, we call them "bad" features
    idx_badFeatures = []
    for d in range(X.shape[1]):
        if sum(X[:,d] == -999) > 0:
            idx_badFeatures.append(d)

    # separate "good" and "bad" features
    X_badFeatures = X[:,idx_badFeatures]
    X_goodFeatures = np.delete(X,(idx_badFeatures), axis=1)

    # Standardize it differently (see : standardize_badFeatures(X))
    tX, mean_x, std_x = my.standardize(X_goodFeatures)
    tX2, mean_x2, std_x2 = standardize_badFeatures(X_badFeatures)

    # comment the 3 next lines if you want to work only with "good" features
    tX = np.hstack((tX, tX2))
    mean_x = np.hstack((mean_x, mean_x2))
    std_x = np.hstack((std_x, std_x2))
    
    return tX, mean_x, std_x

# Now tX already has ones in the first column...
tX, mean_x, std_x = clean_data(X)

## train on the whole dataset
############################

# Select the hyperparameter that have been chosen
final_degree = 10
final_lambda = 0.00239503

# Build the polynomial basis, perform ridge regression
final_X = my.build_poly(tX, final_degree)
final_weights = my.ridge_regression(y, final_X, final_lambda)

## Load test data
##################
DATA_TEST_PATH = 'test.csv'
_, X_test, ids_test = my.load_csv_data(DATA_TEST_PATH)

# Transform test data as it has been done for training : cleaning and polynomial basis
tX_test, mean_xtest, std_xtest = clean_data(X_test)
final_X_test = my.build_poly(tX_test, final_degree)

# Aplly the model to the test data in order to get predictions
OUTPUT_PATH = 'results.csv'
y_pred = my.predict_labels(final_weights, final_X_test)
# create submission file
my.create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

