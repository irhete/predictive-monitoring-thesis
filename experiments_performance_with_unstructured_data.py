"""This script trains, evaluates, and measures the computation times of a predictive model for outcome-oriented predictive process monitoring with structured and unstructured data.

Usage:
  experiments_performance_with_unstructured_data.py <dataset> <method> <text_method> <classifier> <n_iter>

Example:
    experiments_performance_with_unstructured_data.py bpic2012_cancelled single_laststate bong xgboost 5
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import os
import sys
from sys import argv
import pickle
import csv
import time

import cProfile
import pstats

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory
import ClassifierFactory


PARAMS_DIR = "val_results_unstructured"
RESULTS_DIR = "results_performance"


dataset_ref = argv[1]
bucket_enc = argv[2]
text_method_enc = argv[3]
cls_method = argv[4]
n_iter = int(argv[5])

bucket_method, cls_encoding = bucket_enc.split("_")
    
if "_" in text_method_enc:
    text_method, text_enc = text_method_enc.split("_")
else:
    text_method = text_method_enc
    text_enc = cls_encoding
    
method_name = "%s_%s" % (bucket_enc, text_method_enc)

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

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
    
train_ratio = 0.8
random_state = 22
min_cases_for_training = 1

# create results directory
if not os.path.exists(os.path.join(RESULTS_DIR)):
    os.makedirs(os.path.join(RESULTS_DIR))
    
for dataset_name in datasets:
    
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

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    overall_class_ratio = dataset_manager.get_class_ratio(train)
    
    if bucket_enc == "prefix_index":
        nr_eventss = range(min_prefix_length, max_prefix_length+1)
    else:
        nr_eventss = [None]
       
    offline_total_times = []
    online_event_times = []
    text_model_fit_times = []
    
    for ii in range(n_iter):
        print("Starting iteration %s ..." % ii)
        time_train = 0
        time_text_model = 0
    
        # load optimal params
        for nr_events in nr_eventss:
            if nr_events is None:
                optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
            else:
                optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s_%s.pickle" % (cls_method, dataset_name, bucket_enc, nr_events, text_method_enc))

            if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                print("Optimal params for %s %s %s not found" % (cls_method, dataset_name, method_name))
                continue

            with open(optimal_params_filename, "rb") as fin:
                args_all = pickle.load(fin)

            # fit text models and transform for each event
            text_transformer_args = args_all["text_transformer_args"]
            if text_method in ["nb", "bong"]:
                text_transformer_args["nr_selected"] = 500
                if text_method == "nb":
                    text_transformer_args["pos_label"] = dataset_manager.pos_label
            elif text_method in ["pv", "lda"]:
                text_transformer_args["random_seed"] = 22
            if dataset_name in ["github"]:
                text_transformer_args["min_freq"] = 10
            elif dataset_name in ["crm2"]:
                text_transformer_args["min_freq"] = 20

            cls_args = args_all["cls_args"]
            cls_args['n_estimators'] = 500

            start = time.time()
            text_transformer = EncoderFactory.get_encoder(text_method, text_transformer_args=text_transformer_args)
            dt_train_text = text_transformer.fit_transform(train[dataset_manager.static_text_cols+dataset_manager.dynamic_text_cols], 
                                                           train[dataset_manager.label_col])
            time_train += time.time() - start
            time_text_model = time.time() - start

            static_text_cols = []
            dynamic_text_cols = []
            for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
                start = time.time()
                dt_train_text = text_transformer.transform(train[[col]], train[dataset_manager.label_col])
                current_text_cols = ["%s_%s" % (col, text_col) for text_col in dt_train_text.columns]
                dt_train_text.columns = current_text_cols
                train_current = pd.concat([train.drop(col, axis=1), dt_train_text], axis=1, sort=False)
                time_train += time.time() - start
                #del train

                if col in dataset_manager.static_text_cols:
                    static_text_cols.extend(current_text_cols)
                else:
                    dynamic_text_cols.extend(current_text_cols)
                del dt_train_text

            # generate prefixes
            if nr_events is not None:
                start = time.time()
                dt_train_prefixes = dataset_manager.generate_prefix_data(train_current, nr_events, nr_events)
                time_train += time.time() - start
                dt_test_prefixes = dataset_manager.generate_prefix_data(test, nr_events, nr_events)
            else:
                start = time.time()
                dt_train_prefixes = dataset_manager.generate_prefix_data(train_current, min_prefix_length, max_prefix_length)
                time_train += time.time() - start
                dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

            #del test
                
            start = time.time()
            train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
            time_train += time.time() - start
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
            cls = ClassifierFactory.get_classifier(cls_method, cls_args, random_state, min_cases_for_training, overall_class_ratio)

            if cls_method == "svm" or cls_method == "logit":
                pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
            else:
                pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

            start = time.time()
            pipeline.fit(dt_train_prefixes, train_y)
            time_train += time.time() - start
            
            del dt_train_prefixes
            
            # predict separately for each prefix
            test_all_grouped = dt_test_prefixes.groupby(dataset_manager.case_id_col)
            for _, group in test_all_grouped:
                start = time.time()
                for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
                    dt_test_text = text_transformer.transform(group[[col]])
                    current_text_cols = ["%s_%s" % (col, text_col) for text_col in dt_test_text.columns]
                    dt_test_text.columns = current_text_cols
                    group = pd.concat([group.drop(col, axis=1), dt_test_text], axis=1, sort=False)

                pred = pipeline.predict_proba(group)
                time_test = (time.time() - start) / len(group)
                online_event_times.append(time_test)
                
        offline_total_times.append(time_train)
        text_model_fit_times.append(time_text_model)

    offline_total_times = np.array(offline_total_times)
    online_event_times = np.array(online_event_times)
    text_model_fit_times = np.array(text_model_fit_times)
    
    outfile = os.path.join(RESULTS_DIR, "results_performance_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        spamwriter.writerow(["dataset", "bucket_enc", "text_method_enc", "cls", "metric", "score"])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "offline_total_avg", offline_total_times.mean()])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "offline_total_std", offline_total_times.std()])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "online_event_avg", online_event_times.mean()])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "online_event_std", online_event_times.std()])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "offline_text_model_avg", text_model_fit_times.mean()])
        spamwriter.writerow([dataset_name, bucket_enc, text_method_enc, cls_method, "offline_text_model_std", text_model_fit_times.std()])
