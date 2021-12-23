# ------------------------------------------------------------------ #
# Imports
# ------------------------------------------------------------------ #


# Import other libraries
from IPython.display import display
import numpy as np
import pandas as pd
import pyarrow
import os

# Import files
from helpers import *
from cross_validation import *
from models import *
from dataloader import *
from dataprocess import processing


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #


if __name__ == '__main__':

    # Define the classifier
    classifier = 'hydro'

    # Load the data sets
    data_set, classes = load_data_sets(classifier = classifier)

    # Get a processing on the data sets
    X, y = processing(data_set, classes, classifier)

    # Get a train and test set for modelization
    k_fold = 5
    seed = 0
    X_train, y_train, X_test, y_test = split_data(X, y, kfold = k_fold, seed = seed)

    # Feature selection
    method = 'lassoCV'
    model_feat_selec = get_model_features_selection(X_train, y_train, method, k_fold, seed = seed)

    # Select the good features
    X_train_reduce = feature_transform(model_feat_selec, X_train, method)
    X_test_reduce = feature_transform(model_feat_selec, X_test, method)

    # Oversampling
    X_train_reduce, y_train = smote_data_augmentation(X_train_reduce, y_train, seed = seed)

    # Set the verbosity
    verbose = 2

    # MLR model
    MLR, param = get_model_MLR(seed = seed)
    cv_MLR = evaluate_model(MLR, 
                            param, 
                            X_train_reduce, 
                            y_train, 
                            X_test_reduce, 
                            y_test, 
                            verbosity = verbose)

    # Save best model
    path = 'Models/trained_model/'+str(classifier)+'_MLR.pkl'
    save_model(path, cv_MLR)


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #