"""This script optimizes hyperparameters for a deep neural network (with LSTM units) based predictive model for outcome-oriented predictive process monitoring.

***
The architecture of the neural network is based on the approach proposed in the following paper:
Niek Tax, Ilya Verenich, Marcello La Rosa, Marlon Dumas: 
Predictive Business Process Monitoring with LSTM Neural Networks. CAiSE 2017: 477-492,
with code available at: https://github.com/verenich/ProcessSequencePrediction
***

Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import time
import os
import shutil
import sys
from sys import argv
import pickle
import csv
from collections import defaultdict

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

from DatasetManager import DatasetManager

PARAMS_DIR = "val_results_lstm"

# create directory
if not os.path.exists(os.path.join(PARAMS_DIR)):
    os.makedirs(os.path.join(PARAMS_DIR))

def create_and_evaluate_model(args):
    global trial_nr, all_results
    trial_nr += 1
    
    print("Trial %s out of %s" % (trial_nr, n_iter))
    
    model = Sequential()
    
    model.add(CuDNNLSTM(int(params["lstmsize"]),
                           kernel_initializer='glorot_uniform',
                           return_sequences=(n_layers != 1),
                           kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                           recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                           input_shape=(maxlen, num_chars)))
    model.add(Dropout(params["dropout"]))

    for i in range(2, params['n_layers']+1):
        return_sequences = (i != n_layers)
        model.add(CuDNNLSTM(args['lstmsize'],
                       kernel_initializer='glorot_uniform',
                       return_sequences=return_sequences,
                       kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                       recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
        model.add(Dropout(params["dropout"]))
        
    model.add(Dense(2, activation=activation, kernel_initializer='glorot_uniform'))
    #model.add(Activation(tf.nn.sigmoid))
    opt = Adam(lr=params["learning_rate"])
    model.compile(loss='binary_crossentropy', optimizer=opt)
        
        

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    # train the model, output generated text after each iteration
    history = model.fit(X, y,
                        validation_data=(X_val, y_val),
              callbacks=[early_stopping, lr_reducer],
              batch_size=2**params['batch_size'], epochs=nb_epoch, verbose=2)
    
    val_losses = [history.history['val_loss'][epoch] for epoch in range(len(history.history['loss']))]
    #K.clear_session()
    del model
    
    val_losses = val_losses[5:] # don't consider the first few epochs
    best_epoch = np.argmin(val_losses)
    
    # save current trial results
    for k, v in args.items():
        all_results.append((trial_nr, k, v, -1, val_losses[best_epoch]))

    return {'loss': val_losses[best_epoch], 'status': STATUS_OK, 'best_epoch': best_epoch+5}


dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]
n_iter = int(argv[4])

activation = "sigmoid"
nb_epoch = 100
train_ratio = 0.8
val_ratio = 0.2
random_state = 22

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, _ = dataset_manager.split_data_strict(data, train_ratio)
#if "bpic2017" in dataset_name or "traffic" in dataset_name:
#    train, _ = dataset_manager.split_data_strict(train, 0.625) # this makes it use 50% of the original data
train, val = dataset_manager.split_val(train, val_ratio)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
del data
    
dt_train = dataset_manager.encode_data_for_lstm(train)
del train
dt_train = dt_train.sort_values(dataset_manager.timestamp_col, ascending=True, 
                                kind="mergesort").groupby(dataset_manager.case_id_col).head(max_len)
data_dim = dt_train.shape[1] - 3
X, y = dataset_manager.generate_3d_data(dt_train, max_len)
del dt_train

dt_val = dataset_manager.encode_data_for_lstm(val)
del val
dt_val = dt_val.sort_values(dataset_manager.timestamp_col, ascending=True, 
                            kind="mergesort").groupby(dataset_manager.case_id_col).head(max_len)
X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
del dt_val

space = {'lstmsize': scope.int(hp.qloguniform('lstmsize', np.log(10), np.log(150), 1)),
         'dropout': hp.uniform("dropout", 0, 0.3),
         'n_layers': scope.int(hp.quniform('n_layers', 1, 3, 1)),
         'batch_size': scope.int(hp.quniform('batch_size', 3, 6, 1)),
         #'optimizer': hp.choice('optimizer', ["rmsprop", "adam"]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.000001), np.log(0.0001)),
         'l1': hp.loguniform("l1", np.log(0.00001), np.log(0.1)),
         'l2': hp.loguniform("l2", np.log(0.00001), np.log(0.1))}

# optimize parameters
trial_nr = 0
trials = Trials()
all_results = []
best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

# extract the best parameters
best_params = hyperopt.space_eval(space, best)

best_trial_nr = np.argmin([trial['loss'] for trial in trials.results])
best_params['nb_epoch'] = trials.results[best_trial_nr]['best_epoch']

# write to file
outfile = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
with open(outfile, "wb") as fout:
    pickle.dump(best_params, fout)

dt_results = pd.DataFrame(all_results, columns=["iter", "param", "value", "nr_events", "score"])
dt_results["dataset"] = dataset_name
dt_results["cls"] = cls_method
dt_results["method"] = method_name

outfile = os.path.join(PARAMS_DIR, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
dt_results.to_csv(outfile, sep=";", index=False)
