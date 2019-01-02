"""This script trains a deep neural network (with LSTM units) based predictive model for outcome-oriented predictive process monitoring and writes the predictions for a test set to a file.

Usage of the current script:
  python experiments_write_predictions_lstm.py <dataset> <method> <classifier>

Example:
  python experiments_write_predictions_lstm.py bpic2012_cancelled lstm lstm_calibrated
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import time
import os
from sys import argv
import csv
import pickle

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM, Dropout
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from DatasetManager import DatasetManager
from calibration_models import LSTM2D


dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]

PARAMS_DIR = "val_results_lstm"
RESULTS_DIR = "results"
DETAILED_RESULTS_DIR = "%s_detailed" % RESULTS_DIR

optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method.replace("_calibrated", ""),
                                                                                       dataset_name, method_name))
with open(optimal_params_filename, "rb") as fin:
    params = pickle.load(fin)
    
activation = "sigmoid"
nb_epoch = 100
train_ratio = 0.8
val_ratio = 0.2
random_state = 22


##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data_strict(data, train_ratio)
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

if dataset_name != "crm2":
    X, y = dataset_manager.generate_3d_data(dt_train, max_len)
    del dt_train

dt_val = dataset_manager.encode_data_for_lstm(val)
del val
dt_val = dt_val.sort_values(dataset_manager.timestamp_col, ascending=True, 
                            kind="mergesort").groupby(dataset_manager.case_id_col).head(max_len)
    
data_dim = dt_train.shape[1] - 3

X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)

dt_test = dataset_manager.encode_data_for_lstm(test)
dt_test = dt_test.sort_values(dataset_manager.timestamp_col, ascending=True, 
                            kind="mergesort").groupby(dataset_manager.case_id_col).head(max_len)
del test

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

model = Sequential()
    
model.add(CuDNNLSTM(int(params["lstmsize"]),
                       kernel_initializer='glorot_uniform',
                       return_sequences=(params['n_layers'] != 1),
                       kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                       recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                       input_shape=(max_len, data_dim)))
model.add(Dropout(params["dropout"]))

for i in range(2, params['n_layers']+1):
    return_sequences = (i != params['n_layers'])
    model.add(CuDNNLSTM(params['lstmsize'],
                   kernel_initializer='glorot_uniform',
                   return_sequences=return_sequences,
                   kernel_regularizer=regularizers.l1_l2(params["l1"],params["l2"]),
                   recurrent_regularizer=regularizers.l1_l2(params["l1"],params["l2"])))
    model.add(Dropout(params["dropout"]))

model.add(Dense(2, activation=activation, kernel_initializer='glorot_uniform'))
opt = Adam(lr=params["learning_rate"])
model.compile(loss='binary_crossentropy', optimizer=opt)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

# train the model, output generated text after each iteration
if dataset_name == "crm2":
        history = model.fit_generator(dataset_manager.data_generator(dt_train, max_len, 2**params['batch_size']),
                                      validation_data=(X_val, y_val),
                            steps_per_epoch=int(np.ceil(len(dt_train)/2**params['batch_size'])),
              callbacks=[early_stopping, lr_reducer],
                                      epochs=nb_epoch, verbose=2)
else:
    history = model.fit(X, y,
                        validation_data=(X_val, y_val),
              callbacks=[early_stopping, lr_reducer],
              batch_size=2**params['batch_size'], epochs=nb_epoch, verbose=2)

if "calibrate" in cls_method:
    n_cases, time_dim, n_features = X_val.shape
    model_2d = LSTM2D(model, time_dim, n_features)
    model_calibrated = CalibratedClassifierCV(model_2d, cv="prefit", method='sigmoid')
    model_calibrated.fit(X_val.reshape(n_cases, time_dim*n_features), y_val[:,1])

print("Done: %s"%(time.time() - start))


# Write loss for each epoch
print('Evaluating...')
start = time.time()

detailed_results = pd.DataFrame()
detailed_results_val = pd.DataFrame()
for nr_events in range(1, max_len+1):
    # encode only prefixes of this length
    X, y, case_ids = dataset_manager.generate_3d_data_for_prefix_length(dt_test, max_len, nr_events)

    if X.shape[0] == 0:
        break

    if "calibrate" in cls_method:
        preds = model_calibrated.predict_proba(X.reshape(X.shape[0], time_dim*n_features))[:,1]
    else:
        preds = model.predict(X, verbose=0)[:,1]

    current_results = pd.DataFrame({"dataset": dataset_name, "method": method_name, "cls": cls_method,
                                    "nr_events": nr_events, "predicted": preds, "actual": y[:,1], "case_id": case_ids})
    detailed_results = pd.concat([detailed_results, current_results], axis=0, sort=False)
    
    if "calibrate" in cls_method:
        X, y, case_ids = dataset_manager.generate_3d_data_for_prefix_length(dt_val, max_len, nr_events)
        preds = model_calibrated.predict_proba(X.reshape(X.shape[0], time_dim*n_features))[:,1]
        current_results = pd.DataFrame({"dataset": dataset_name, "method": method_name, "cls": cls_method,
                                    "nr_events": nr_events, "predicted": preds, "actual": y[:,1], "case_id": case_ids})
        detailed_results_val = pd.concat([detailed_results_val, current_results], axis=0, sort=False)
    
print("Done: %s"%(time.time() - start))
        
# Write predictions
detailed_results_file = os.path.join(DETAILED_RESULTS_DIR, "detailed_results_test_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
detailed_results.to_csv(detailed_results_file, sep=";", index=False)
if "calibrate" in cls_method:
    detailed_results_file = os.path.join(DETAILED_RESULTS_DIR, "detailed_results_val_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
    detailed_results_val.to_csv(detailed_results_file, sep=";", index=False)
