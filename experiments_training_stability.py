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
import time

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


dataset_name = argv[1]
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

#datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
overall_train_ratio = 0.8
random_state = 22
min_cases_for_training = 1
    
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
if dataset_name in ["github", "crm2", "dc"]:
    train_all, _ = dataset_manager.split_data(data, overall_train_ratio, split="random", seed=22)
else:
    train_all, _ = dataset_manager.split_data_strict(data, overall_train_ratio, split="temporal")

train_chunks = dataset_manager.split_chunks(train_all, n_chunks=10)

for n_train_chunks in range(1, 10):
    for n_test_chunks in range(1, 10-n_train_chunks+1):
        train = pd.concat(train_chunks[:n_train_chunks], axis=0, sort=False)
        test = pd.concat(train_chunks[-n_test_chunks:], axis=0, sort=False)

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
            if cls_method == "rf":
                cls = RandomForestClassifier(n_estimators=500,
                                                        random_state=random_state,
                                                        n_jobs=-1)
            elif cls_method == "xgboost":
                cls = xgb.XGBClassifier(objective='binary:logistic',
                                  n_estimators=500,
                                  n_jobs=-1)
                
            if cls_method == "svm" or cls_method == "logit":
                pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
            else:
                pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

            # fit pipeline
            start = time.time()
            pipeline.fit(dt_train_bucket, train_y)
            train_time = time.time() - start
            
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0] 
            
            # predict 
            start = time.time()
            preds = pipeline.predict_proba(dt_test_bucket)[:,preds_pos_label_idx]
            test_time = time.time() - start
            preds_all.extend(preds)

        auc = roc_auc_score(test_y_all, preds_all)
        print(n_train_chunks, n_test_chunks, auc, train_time, test_time)
        