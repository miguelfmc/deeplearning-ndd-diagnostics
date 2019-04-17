""" Train and evaluate one classifier for each class
"""

import os
import csv

from tensorflow.keras import optimizers
from .model_utils import load_dataset, save_model, load_model, redefine_labels, standardize_per_example
from .metrics import precision, recall, f1_score

CLASSES = {
    'control': 0,
    'als': 1,
    'hunt': 2,
    'park': 3
}

def train_disease(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, get_model, disease, hyperparameters, dataset_name):
    """
    Trains a binary classifier (keras model) on the given data for the given disease (class)

    Arguments
        X_train: an ndarray with the training data input features
        Y_train: an ndarray with the target labels in multiclass mode
        X_dev: an ndarray with the dev data input features
        Y_dev: an ndarray with the dev data target labels
        X_test: an ndarray with the test data input features
        Y_test: an ndarrat with the test data target labels
        get_model: handle of the function that returns the keras model
            that will be trained
        disease: string, name of the disease to classify
        hyperparameters: tuple with (learning_rate, n_epochs, batch_size)
        dataset_name: one of 'spectrograms-dataset', 'scalograms-dataset' or 'signals-dataset'
    
    Returns
        model: keras model
        history: training history
        train_evaluations: list of metrics evaluated on the train set (after training has finished)
        dev_evaluations: list of metrics evaluated on the dev set
        test_evalations: list of metrics evaluated on the test set

    """

    target_label = CLASSES[disease]
    
    # redefine labels
    Y_train = redefine_labels(Y_train, target_label)
    Y_dev = redefine_labels(Y_dev, target_label)
    Y_test =redefine_labels(Y_test, target_label)
    
    # get hparams
    learning_rate, n_epochs, batch_size = hyperparameters

    # build model and train
    model = get_model(X_train.shape[1:])
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', precision, recall, f1_score])
    
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs,
                        validation_data=(X_dev, Y_dev))
    
    # save model
    model_dir = os.path.join('trained_models', dataset_name.replace('-dataset', ''))
    model_name = get_model.__name__
    name = model_name + '_' + disease
    
    save_model(model, history, name, model_dir)
    
    # evaluate on dev and test
    train_evaluations = model.evaluate(X_train, Y_train)
    dev_evaluations = model.evaluate(X_dev, Y_dev)
    test_evaluations = model.evaluate(X_test, Y_test)

    # save results to CSV
    csv_path = os.path.join('trained_models', 'results.csv')
    
    fields = [model_name, disease, dataset_name, n_epochs, learning_rate, batch_size]
    fields.extend(train_evaluations)
    fields.extend(dev_evaluations)
    fields.extend(test_evaluations)
    fields.append(name)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
    return model, history, train_evaluations, dev_evaluations, test_evaluations

def train(dataset_name, get_model, hyperparameters):
    """
    Trains all four classifiers (one per disease) with a given architecture and hyperparameters

    Arguments
        dataset_name: one of 'spectrograms-dataset', 'scalograms-dataset' or 'signals-dataset'
        get_model: handle to the function that returns the keras model that will be trained
        hyperparameters: tuple with (learning_rate, n_epochs, batch_size) 
    """
    # load data
    (X_train_orig, Y_train_orig, Z_train_orig, X_dev_orig,
    Y_dev_orig, Z_dev_orig, X_test_orig, Y_test_orig, Z_test_orig) = load_dataset(dataset_name, mix=True)

    # standardize
    if dataset_name == 'signals-dataset':
        X_train = X_train_orig
        X_dev = X_dev_orig
        X_test = X_test_orig
    else:
        X_train = standardize_per_example(X_train_orig)
        X_dev = standardize_per_example(X_dev_orig)
        X_test = standardize_per_example(X_test_orig)

    Y_train = Y_train_orig
    Y_dev = Y_dev_orig
    Y_test = Y_test_orig

    # run training and evaluation for each disease
    diseases = ['als', 'hunt', 'park', 'control']
    for disease in diseases:
        train_disease(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, get_model, disease, hyperparameters, dataset_name)