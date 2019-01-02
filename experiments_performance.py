"""This script trains, evaluates, and measures the computation times of a predictive model for outcome-oriented predictive process monitoring.

Usage:
  experiments_performance.py <dataset> <method> <classifier> <n_iter>

Example:
    experiments_performance.py bpic2012_cancelled single_laststate xgboost 5
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import time
import os
import sys
from sys import argv
import pickle
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory
import ClassifierFactory

PARAMS_DIR = "cv_results_revision"
#PARAMS_DIR = "val_results_unstructured"
RESULTS_DIR = "results_performance"

dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]
n_iter = int(argv[4])

gap = 1

bucket_method, cls_encoding = method_name.split("_")


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
    
    if bucket_method != "prefix":
        # load optimal params
        optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
        if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
            continue

        with open(optimal_params_filename, "rb") as fin:
            args = pickle.load(fin)
    
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
    
    # generate test prefix log
    start_test_prefix_generation = time.time()
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    test_prefix_generation_time = time.time() - start_test_prefix_generation
            
    offline_total_times = []
    online_event_times = []
    
    # multiple iterations for performance calculations
    for ii in range(n_iter):
        print("Starting iteration %s ..." % ii)
        time_train = 0
        
        # create train prefix log
        start = time.time()
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
        time_train += time.time() - start
            
        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding, 
                         'case_id_col': dataset_manager.case_id_col, 
                         'cat_cols': [dataset_manager.activity_col], 
                         'num_cols': [], 
                         'random_state': random_state}
        if bucket_method == "cluster":
            bucketer_args["n_clusters"] = int(args["n_clusters"])
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

        start = time.time()
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        time_train += time.time() - start

        bucket_assignments_test = bucketer.predict(dt_test_prefixes)

        for bucket in set(bucket_assignments_test):
            if bucket_method == "prefix":
                # load optimal params
                optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name, bucket))
                if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                    continue

                with open(optimal_params_filename, "rb") as fin:
                    args = pickle.load(fin)
            
            args["n_estimators"] = 500
            
            # select prefixes for the given bucket
            relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
            dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            test_y = dataset_manager.get_label_numeric(dt_test_bucket)
            
            # initialize pipeline for sequence encoder and classifier
            start = time.time()
            feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            cls = ClassifierFactory.get_classifier(cls_method, args, random_state, min_cases_for_training, 
                                                   overall_class_ratio)

            if cls_method == "svm" or cls_method == "logit":
                pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
            else:
                pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

            # fit pipeline
            pipeline.fit(dt_train_bucket, train_y)
            time_train += time.time() - start

            # predict separately for each prefix
            test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
            for _, group in test_all_grouped:
                start = time.time()
                _ = bucketer.predict(group)
                pred = pipeline.predict_proba(group)
                time_test = (time.time() - start) / len(group)
                online_event_times.append(time_test)
        offline_total_times.append(time_train)

    offline_total_times = np.array(offline_total_times)
    online_event_times = np.array(online_event_times)
    
    # write results
    outfile = os.path.join(RESULTS_DIR, "results_performance_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        spamwriter.writerow(["dataset", "bucket_enc", "text_method_enc", "cls", "metric", "score"])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "offline_total_avg", offline_total_times.mean()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "offline_total_std", offline_total_times.std()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "online_event_avg", online_event_times.mean()])
        spamwriter.writerow([dataset_name, method_name, "no_text", cls_method, "online_event_std", online_event_times.std()])
