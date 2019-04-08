"""Useful functions to interact with tensorflow.keras models and with the training and testing dataset
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import pickle


def load_dataset(dataset_name='spectrograms-dataset', mix=True):
    """Loads one of the datasets: spectrograms or scalograms,
    either with mix of patients between the train and dev sets or not

    Arguments
        dataset_name: string defining the dataset
        mix: boolean that indicates if a mixed train-dev set should be loaded or not
            if set to True records from the same patients appear in both training and dev set
            if set to False 4 the dev set will be comprised of records from 4 patients that
                will not appear in the train set
    
    Returns
        X (images), Y (labels), Z (other info) of train, dev and test datasets as numpy ndarrays
    """
    in_dir = os.path.join('data', 'processed', dataset_name)

    if mix:
        train_data = np.load(os.path.join(in_dir, 'train-dev.npz'))

        X_orig = train_data['X_train']
        Y_orig = train_data['Y_train']
        Z_orig = train_data['Z_train']

        X_orig = X_orig.reshape(X_orig.shape[0], X_orig.shape[1], X_orig.shape[2], 1)

        # random state seed to always keep the same split
        X_train_orig, X_dev_orig, Y_train_orig, Y_dev_orig, Z_train_orig, Z_dev_orig = train_test_split(
            X_orig, Y_orig, Z_orig, random_state=1, test_size=0.20)
    else:
        train_data = np.load(os.path.join(in_dir, 'train.npz'))
        dev_data = np.load(os.path.join(in_dir, 'dev.npz'))

        X_train_orig = train_data['X_train']
        Y_train_orig = train_data['Y_train']
        Z_train_orig = train_data['Z_train']

        X_train_orig = X_train_orig.reshape(X_train_orig.shape[0], X_train_orig.shape[1], X_train_orig.shape[2], 1)

        X_dev_orig = dev_data['X_dev']
        Y_dev_orig = dev_data['Y_dev']
        Z_dev_orig = dev_data['Z_dev']

        X_dev_orig = X_dev_orig.reshape(X_dev_orig.shape[0], X_dev_orig.shape[1], X_dev_orig.shape[2], 1)

    # test data always the same
    test_data = np.load(os.path.join(in_dir, 'test.npz'))

    X_test_orig = test_data['X_test']
    Y_test_orig = test_data['Y_test']
    Z_test_orig = test_data['Z_test']

    X_test_orig = X_test_orig.reshape(X_test_orig.shape[0], X_test_orig.shape[1], X_test_orig.shape[2], 1)

    return X_train_orig, Y_train_orig, Z_train_orig, X_dev_orig, Y_dev_orig, Z_dev_orig, X_test_orig, Y_test_orig, Z_test_orig


def save_model(model, history, name):
    """
    Saves trained model architecture, history and weights as a JSON file with given name
    """
    out_dir = 'trained_models'

    # save model architecture
    model_json = model.to_json()
    model_path = os.path.join(out_dir, name + '_arch.json')
    with open(model_path, 'w') as json_file:
        json_file.write(model_json)
    
    # save model weigths (saving entire model is problematic)
    weights_path = os.path.join(out_dir, name + '_weights.h5')
    model.save_weights(weights_path)
    
    # save history
    history_dict = history.history
    history_path = os.path.join(out_dir, '_'.join([name, 'hist.pickle']))
    with open(history_path, 'wb') as handle:
        pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return None


def load_model(name):
    """
    Loads trained model with weights as well as its training history
    """
    in_dir = 'trained_models'
    
    # load model architecture
    model_path = os.path.join(in_dir, name + '_arch.json')
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    
    # load model weigths
    weights_path = os.path.join(in_dir, name + '_weights.h5')
    model.load_weights(weights_path)
    
    # load history
    history_path = os.path.join(in_dir, name + '_hist.pickle')
    with open(history_path, 'rb') as handle:
        history = pickle.load(handle)
    
    return model, history


def redefine_labels(Y, target_class):
    """
    Transforms a K-labeled vector into a binary label vector. If Y[i] is equal to target_class
        then Y_binary[i] is 1. Otherwise, Y[i] is 0
    
    Arguments
        Y: a (m, 1) NumPy array of labels ranging from 0 to (K - 1) with K being the number of classes
        target_class: integer indicating the target class
    
    Returns
        Y_binary: a binary labelled (0 or 1) NumPy array of the same shape as Y
    """
    
    Y_binary = (Y == target_class).astype(int)
    
    return Y_binary


def standardize_per_example(X):
    """
    Standardizes a numpy ndarray by dividing each example by its range

    Arguments
        X: numpy ndarray of dimensions (m, h, w, c)
    
    Returns:
        The standardized array of same dimensions
    """
    maxes = X.max(axis=(1,2,3), keepdims=True)
    mins = X.min(axis=(1,2,3), keepdims=True)
    
    return X / (maxes - mins)