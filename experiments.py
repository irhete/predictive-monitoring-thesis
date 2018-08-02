"""This script trains and evaluates a predictive model for outcome-oriented predictive process monitoring.

Usage:
  experiments.py <dataset> <method> <classifier>

Example:
    experiments.py bpic2012_cancelled single_laststate xgboost
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

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
RESULTS_DIR = "results"

dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]

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
    
    # generate prefix logs
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
            
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
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    bucket_assignments_test = bucketer.predict(dt_test_prefixes)

    preds_all = []
    test_y_all = []
    nr_events_all = []
    for bucket in set(bucket_assignments_test):
        current_args = args if bucket_method != "prefix" else args[bucket]
        current_args["n_estimators"] = 500
            
        # select prefixes for the given bucket
        relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
            
        # add data about prefixes in this bucket (class labels and prefix lengths)
        nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
        test_y_all.extend(test_y)

        # initialize pipeline for sequence encoder and classifier
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        cls = ClassifierFactory.get_classifier(cls_method, current_args, random_state, min_cases_for_training, overall_class_ratio)

        if cls_method == "svm" or cls_method == "logit":
            pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
        else:
            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

        # fit pipeline
        pipeline.fit(dt_train_bucket, train_y)

        # predict 
        preds = pipeline.predict_proba(dt_test_bucket)
        preds_all.extend(preds)

    # write results
    outfile = os.path.join(RESULTS_DIR, "results_%s_%s_%s_gap%s.csv" % (cls_method, dataset_name, method_name, gap))
    with open(outfile, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        spamwriter.writerow(["dataset", "method", "cls", "nr_events", "metric", "score"])

        dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
        for nr_events, group in dt_results.groupby("nr_events"):
            auc = np.nan if len(set(group.actual)) < 2 else roc_auc_score(group.actual, group.predicted)
            spamwriter.writerow([dataset_name, method_name, cls_method, nr_events, -1, "auc", auc])
            print(nr_events, auc)

        auc = roc_auc_score(dt_results.actual, dt_results.predicted)
        spamwriter.writerow([dataset_name, method_name, cls_method, -1, "auc", auc])
        print(nr_events, auc)
