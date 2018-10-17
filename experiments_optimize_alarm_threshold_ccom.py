import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle
import csv

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt

PREDS_DIR = "predictions"
PARAMS_DIR = "optimal_alarm_thresholds_ccom"

def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def evaluate_model_cost(args):
    conf_threshold = args['conf_threshold']
    
    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
        tmp = tmp[tmp.predicted_proba >= conf_threshold]
        tmp["prediction"] = 1
        dt_final = pd.concat([dt_final, tmp], axis=0, sort=False)
        unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
    tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
    tmp["prediction"] = 0
    dt_final = pd.concat([dt_final, tmp], axis=0, sort=False)

    case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
    case_lengths.columns = ["case_id", "case_length"]
    dt_final = dt_final.merge(case_lengths)
        
    cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()
    
    return {'loss': cost, 'status': STATUS_OK, 'model': dt_final}


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]

# create output directory
if not os.path.exists(os.path.join(RESULTS_DIR)):
    os.makedirs(os.path.join(RESULTS_DIR))
    
# read the data
dataset_manager = DatasetManager(dataset_name)
    
# prepare the dataset
dt_preds = pd.read_csv(os.path.join(PREDS_DIR, "preds_val_%s_%s_%s.csv" % (dataset_name, method_name, cls_method)), sep=";")

print('Optimizing parameters...')
cost_weights = [(1,1), (2,1), (3,1), (5,1), (10,1), (20,1), (40, 1)]
c_com_weights = [1/40.0, 1/20.0, 1/10.0, 1/5.0, 1/3.0, 1/2.0, 1, 2, 3, 5, 10, 20, 40, 0]
c_postpone_weight = 0
for c_miss_weight, c_action_weight in cost_weights:
    for c_com_weight in c_com_weights:
        for early_type in ["const", "linear"]:
    
            c_miss = c_miss_weight / (c_miss_weight + c_action_weight + c_com_weight)
            c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
            c_com = c_com_weight / (c_miss_weight + c_action_weight + c_com_weight)

            if early_type == "linear":
                costs = np.matrix([[lambda x: 0,
                                      lambda x: c_miss],
                                     [lambda x: c_action * (x['prefix_nr']-1) / x['case_length'] + c_com,
                                      lambda x: c_action * (x['prefix_nr']-1) / x['case_length'] + (x['prefix_nr']-1) / x['case_length'] * c_miss
                                     ]])
            else:
                costs = np.matrix([[lambda x: 0,
                                      lambda x: c_miss],
                                     [lambda x: c_action + c_com,
                                      lambda x: c_action + (x['prefix_nr']-1) / x['case_length'] * c_miss
                                     ]])

            space = {'conf_threshold': hp.uniform("conf_threshold", 0, 1)}
            trials = Trials()
            best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=50, trials=trials)

            best_params = hyperopt.space_eval(space, best)

            outfile = os.path.join(PARAMS_DIR, "optimal_confs_%s_%s_%s_%s_%s_%s_%s_%s.pickle" % (dataset_name, method_name, 
                                                                                           cls_method, c_miss_weight, 
                                                                                           c_action_weight, c_postpone_weight, 
                                                                                           c_com_weight, early_type))
            # write to file
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
