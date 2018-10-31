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

PARAMS_DIR = "val_results_unstructured"

# create directory
if not os.path.exists(os.path.join(PARAMS_DIR)):
    os.makedirs(os.path.join(PARAMS_DIR))

def create_and_evaluate_model(args):
    global trial_nr, all_results
    trial_nr += 1
    
    print("Trial %s out of %s" % (trial_nr, n_iter))
    
    start = time.time()
    score = 0
    
    cls_args = {k: v for k, v in args.items() if k in cls_params}
    text_transformer_args = {k: v for k, v in args.items() if k not in cls_params}
    cls_args['n_estimators'] = 500
    
    for cv_iter in range(n_splits):
        
        # read encoded data
        train_chunk = dataset_manager.read_fold(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter))
        test_chunk = dataset_manager.read_fold(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter))
        
        # fit text models and transform for each event
        if text_method in ["nb", "bong"]:
            text_transformer_args["nr_selected"] = 500
            if text_method == "nb":
                text_transformer_args["pos_label"] = dataset_manager.pos_label
        elif text_method in ["pv", "lda"]:
            text_transformer_args["random_seed"] = 22
        if dataset_name in ["github"]:
            text_transformer_args["min_freq"] = 10
        elif dataset_name in ["crm2"]:
            text_transformer_args["min_freq"] = 10
        
        text_transformer = EncoderFactory.get_encoder(text_method, text_transformer_args=text_transformer_args)
        dt_train_text = text_transformer.fit_transform(train_chunk[dataset_manager.static_text_cols+dataset_manager.dynamic_text_cols], 
                                                       train_chunk[dataset_manager.label_col])
        
        static_text_cols = []
        dynamic_text_cols = []
        for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
            dt_train_text = text_transformer.transform(train_chunk[[col]], train_chunk[dataset_manager.label_col])
            current_text_cols = ["%s_%s" % (col, text_col) for text_col in dt_train_text.columns]
            dt_train_text.columns = current_text_cols
            dt_test_text = text_transformer.transform(test_chunk[[col]])
            dt_test_text.columns = current_text_cols
            train_chunk = pd.concat([train_chunk.drop(col, axis=1), dt_train_text], axis=1, sort=False)
            test_chunk = pd.concat([test_chunk.drop(col, axis=1), dt_test_text], axis=1, sort=False)
            if col in dataset_manager.static_text_cols:
                static_text_cols.extend(current_text_cols)
            else:
                dynamic_text_cols.extend(current_text_cols)
            del dt_train_text, dt_test_text
        
        # generate prefixes
        if nr_events is not None:
            dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, nr_events, nr_events)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, nr_events, nr_events)
        else:
            dt_train_prefixes = dataset_manager.generate_prefix_data(train_chunk, min_prefix_length, max_prefix_length)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)
                
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
            
        # set up sequence encoders
        encoders = []
        for method in methods:
            if cls_encoding == text_enc:
                cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols + static_text_cols, 
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols + dynamic_text_cols, 
                    'fillna': True}
            else:
                cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols + static_text_cols, 
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                    'fillna': True}
            encoders.append((method, EncoderFactory.get_encoder(method, **cls_encoder_args)))
        if cls_encoding != text_enc and text_enc not in methods:
            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                    'static_cat_cols': [],
                    'static_num_cols': [], 
                    'dynamic_cat_cols': [],
                    'dynamic_num_cols': dynamic_text_cols, 
                    'fillna': True}
            encoders.append((text_enc, EncoderFactory.get_encoder(text_enc, **cls_encoder_args)))
                
        feature_combiner = FeatureUnion(encoders)
        
        # fit classifier and predict
        cls = ClassifierFactory.get_classifier(cls_method, cls_args, random_state, min_cases_for_training, class_ratios[cv_iter])

        if cls_method == "svm" or cls_method == "logit":
            pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
        else:
            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

        pipeline.fit(dt_train_prefixes, train_y)
        preds = pipeline.predict_proba(dt_test_prefixes)

        if len(set(test_y)) >= 2:
            score += roc_auc_score(test_y, preds)
    
    # save current trial results
    for k, v in cls_args.items():
        all_results.append((trial_nr, k, v, -1, score / n_splits))
    for k, v in text_transformer_args.items():
        all_results.append((trial_nr, k, v, -1, score / n_splits))

    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


dataset_ref = argv[1]
bucket_enc = argv[2]
text_method_enc = argv[3]
cls_method = argv[4]
n_iter = int(argv[5])

train_ratio = 0.8
n_splits = 1
random_state = 22
min_cases_for_training = 1

if "prefix_index" in bucket_enc:
    bucket_method, cls_encoding, nr_events = bucket_enc.split("_")
    nr_events = int(nr_events)
else:
    bucket_method, cls_encoding = bucket_enc.split("_")
    nr_events = None
    
if "_" in text_method_enc:
    text_method, text_enc = text_method_enc.split("_")
else:
    text_method = text_method_enc
    text_enc = cls_encoding

method_name = "%s_%s" % (bucket_enc, text_method_enc)

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
    
    # determine min and max (truncated) prefix lengths
    min_prefix_length = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    del data
    
    # prepare chunks for CV
    class_ratios = []
    cv_iter = 0
    if n_splits == 1:
        if dataset_ref in ["github"]:
            train, _ = dataset_manager.split_data(train, train_ratio=0.15/train_ratio, split="random", seed=22)
            # train will be 0.1 of original data and val 0.05
            train_chunk, test_chunk = dataset_manager.split_val(train, val_ratio=0.33, split="random", seed=22)
        else:
            train_chunk, test_chunk = dataset_manager.split_val(train, 0.2, split="random", seed=22)
        pd.DataFrame(train_chunk).to_csv(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter), sep=";", index=False)
        pd.DataFrame(test_chunk).to_csv(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter), sep=";", index=False)
        class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
    
    else:
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
            class_ratios.append(dataset_manager.get_class_ratio(train_chunk))

            pd.DataFrame(train_chunk).to_csv(os.path.join(folds_dir, "fold%s_train.csv" % cv_iter), sep=";", index=False)
            pd.DataFrame(test_chunk).to_csv(os.path.join(folds_dir, "fold%s_test.csv" % cv_iter), sep=";", index=False)

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
    
    cls_params = set(space.keys())
    
    if text_method == "bong":
        #if dataset_name in ["crm2", "github"]:
        #    space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 1, 1))
        #else:
        if dataset_name == "crm2":
            space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 2, 1))
        else:
            space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 3, 1))
        #space["nr_selected"] = scope.int(hp.qloguniform('nr_selected', np.log(100), np.log(5000), 1))
        space["tfidf"] = hp.choice('tfidf', [True, False])
        
    elif text_method == "nb":
        #if dataset_name in ["crm2", "github"]:
        #    space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 1, 1))
        #else:
        if dataset_name == "crm2":
            space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 2, 1))
        else:
            space["ngram_max"] = scope.int(hp.quniform('ngram_max', 1, 3, 1))
        #space["nr_selected"] = scope.int(hp.qloguniform('nr_selected', np.log(100), np.log(5000), 1))
        space["alpha"] = hp.loguniform('alpha', np.log(0.01), np.log(1))
        
    elif text_method == "lda":
        space["num_topics"] = scope.int(hp.qloguniform('num_topics', np.log(10), np.log(200), 1))
        space["tfidf"] = hp.choice('tfidf', [True, False])
        
    elif text_method == "pv":
        space["size"] = scope.int(hp.qloguniform('size', np.log(10), np.log(400), 1))
        space["window"] = scope.int(hp.quniform('window', 1, 13, 1))
        
    # optimize parameters
    trial_nr = 0
    trials = Trials()
    all_results = []
    best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)

    # extract the best parameters
    best_params = hyperopt.space_eval(space, best)
    
    cls_args = {k: v for k, v in best_params.items() if k in cls_params}
    text_args = {k: v for k, v in best_params.items() if k not in cls_params}
    best_params = {'cls_args': cls_args, 'text_transformer_args': text_args}
    
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