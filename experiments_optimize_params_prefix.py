import time
import os
import shutil
import sys
from sys import argv
import pickle
import csv
from collections import defaultdict

import cProfile
import pstats

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

from DatasetManager import DatasetManager
import EncoderFactory
import ClassifierFactory


PARAMS_DIR = "cv_results"

# create directory
if not os.path.exists(os.path.join(PARAMS_DIR)):
    os.makedirs(os.path.join(PARAMS_DIR))

def create_and_evaluate_model(args):
    global trial_nr, all_results
    trial_nr += 1
    
    print("Trial %s out of %s" % (trial_nr, n_iter))
    
    start = time.time()
    score = 0
    args['n_estimators'] = 500
    scores = defaultdict(int)

    for cv_iter in range(n_splits):
            
        preds_all = []
        test_y_all = []
        for nr_events in range(min_prefix_length, max_prefix_length+1):
            # read encoded data
            dt_train_prefixes = pd.read_csv(os.path.join(folds_dir, "fold%s_%s_train.csv" % (cv_iter, nr_events)), sep=";")
            dt_test_prefixes = pd.read_csv(os.path.join(folds_dir, "fold%s_%s_test.csv" % (cv_iter, nr_events)), sep=";")

            with open(os.path.join(folds_dir, "fold%s_%s_train_y.csv" % (cv_iter, nr_events)), "rb") as fin:
                train_y = np.array(pickle.load(fin))
            with open(os.path.join(folds_dir, "fold%s_%s_test_y.csv" % (cv_iter, nr_events)), "rb") as fin:
                test_y = np.array(pickle.load(fin))
            
            # fit classifier and predict
            cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training,
                                                   class_ratios[cv_iter])
            cls.fit(dt_train_prefixes, train_y)
            preds = cls.predict_proba(dt_test_prefixes)

            # calculate AUC for each prefix
            auc = 0.5
            if len(set(test_y)) == 2: 
                auc = roc_auc_score(test_y, preds)
            scores[nr_events] += auc
            print(nr_events, auc)
                
            preds_all.extend(preds)
            test_y_all.extend(test_y)

        score += roc_auc_score(test_y_all, preds_all)
    
    # save current trial results
    for k, v in args.items():
        for nr_events, bucket_score in scores.items():
            all_results.append((trial_nr, k, v, nr_events, bucket_score / n_splits))
            print(bucket_score / n_splits)
        all_results.append((trial_nr, k, v, -1, score / n_splits))
            
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]
n_iter = int(argv[4])

train_ratio = 0.8
n_splits = 3
random_state = 22
min_cases_for_training = 1

bucket_method, cls_encoding = method_name.split("_")

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
for dataset_name in datasets:
    
    folds_dir = "folds_%s_%s_%s" % (dataset_name, cls_method, method_name)
    if not os.path.exists(os.path.join(folds_dir)):
        os.makedirs(os.path.join(folds_dir))
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    # split into training and test
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    # prepare chunks for CV
    class_ratios = []
    cv_iter = 0
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
        
        # generate prefixes
        dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)
        
        for nr_events in range(min_prefix_length, max_prefix_length+1):

            # encode data for classifier
            feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            if cls_method == "svm" or cls_method == "logit":
                feature_combiner = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler())])

            dt_train_encoded = feature_combiner.fit_transform(dt_train_prefixes[dt_train_prefixes.prefix_nr == nr_events])
            pd.DataFrame(dt_train_encoded).to_csv(os.path.join(folds_dir, "fold%s_%s_train.csv" % (cv_iter, nr_events)), sep=";", index=False)
            del dt_train_encoded

            dt_test_encoded = feature_combiner.transform(dt_test_prefixes[dt_test_prefixes.prefix_nr == nr_events])
            pd.DataFrame(dt_test_encoded).to_csv(os.path.join(folds_dir, "fold%s_%s_test.csv" % (cv_iter, nr_events)), sep=";", index=False)
            del dt_test_encoded

            # labels
            with open(os.path.join(folds_dir, "fold%s_%s_train_y.csv" % (cv_iter, nr_events)), "wb") as fout:
                pickle.dump(dataset_manager.get_label_numeric(dt_train_prefixes[dt_train_prefixes.prefix_nr == nr_events]), fout)

            with open(os.path.join(folds_dir, "fold%s_%s_test_y.csv" % (cv_iter, nr_events)), "wb") as fout:
                pickle.dump(dataset_manager.get_label_numeric(dt_test_prefixes[dt_test_prefixes.prefix_nr == nr_events]), fout)

        cv_iter += 1

    del train
        
    # set up search space
    if cls_method == "rf":
        space = {'max_features': hp.uniform('max_features', 0, 1)}
        
    elif cls_method == "xgboost":
        space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
        
    elif cls_method == "logit":
        space = {'C': hp.uniform('C', -15, 15)}
        
    elif cls_method == "svm":
        space = {'C': hp.uniform('C', -15, 15),
                 'gamma': hp.uniform('gamma', -15, 15)}
        
    # optimize parameters
    trial_nr = 0
    trials = Trials()
    all_results = []
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

    # extract the best parameters
    best_params = hyperopt.space_eval(space, best)
    
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

    shutil.rmtree(folds_dir)