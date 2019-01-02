"""This script trains a predictive model for outcome-oriented predictive process monitoring and writes the predictions to a file.

Usage:
    experiments_write_predictions_alarms_unstructured.py <dataset> <method> <classifier>

Example:
    experiments_write_predictions_alarms_unstructured.py bpic2012_cancelled single_laststate xgboost
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import os
import sys
from sys import argv
import pickle
import csv

import cProfile
import pstats

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory
import ClassifierFactory


PARAMS_DIR = "val_results_unstructured"
PREDS_DIR = "predictions"

dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]

gap = 1

bucket_method, cls_encoding = method_name.split("_")


if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"
    
if "single" in method_name and "index" not in method_name:
    text_method_enc = "bong"
    text_method = text_method_enc
    text_enc = cls_encoding
else:
    text_method_enc = "bong_agg"
    text_method, text_enc = text_method_enc.split("_")

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
val_ratio = 0.2
random_state = 22
min_cases_for_training = 1

# create results directory
if not os.path.exists(os.path.join(PREDS_DIR)):
    os.makedirs(os.path.join(PREDS_DIR))
    
for dataset_name in datasets:
    
    # load optimal params
    optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method.replace("_calibrated", ""), dataset_name, method_name, text_method_enc))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue
        
    with open(optimal_params_filename, "rb") as fin:
        args_all = pickle.load(fin)
    
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
    train, val = dataset_manager.split_val(train, val_ratio)
    overall_class_ratio = dataset_manager.get_class_ratio(train)
    
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

    text_transformer = EncoderFactory.get_encoder(text_method, text_transformer_args=text_transformer_args)
    dt_train_text = text_transformer.fit_transform(train[dataset_manager.static_text_cols+dataset_manager.dynamic_text_cols], 
                                                   train[dataset_manager.label_col])

    static_text_cols = []
    dynamic_text_cols = []
    for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
        dt_train_text = text_transformer.transform(train[[col]], train[dataset_manager.label_col])
        current_text_cols = ["%s_%s" % (col, text_col) for text_col in dt_train_text.columns]
        dt_train_text.columns = current_text_cols
        dt_test_text = text_transformer.transform(test[[col]])
        dt_test_text.columns = current_text_cols
        train_current = pd.concat([train.drop(col, axis=1), dt_train_text], axis=1, sort=False)
        test_current = pd.concat([test.drop(col, axis=1), dt_test_text], axis=1, sort=False)
        if col in dataset_manager.static_text_cols:
            static_text_cols.extend(current_text_cols)
        else:
            dynamic_text_cols.extend(current_text_cols)
        del dt_train_text, dt_test_text
        if "single" in method_name:
            del train, test
        dt_val_text = text_transformer.transform(val[[col]])
        dt_val_text.columns = current_text_cols
        val_current = pd.concat([val.drop(col, axis=1), dt_val_text], axis=1, sort=False)
        del dt_val_text
        if "single" in method_name:
            del val

    
    # generate prefix logs
    dt_test_prefixes = dataset_manager.generate_prefix_data(test_current, min_prefix_length, max_prefix_length)
    del test_current
    dt_val_prefixes = dataset_manager.generate_prefix_data(val_current, min_prefix_length, max_prefix_length)
    del val_current
    dt_train_prefixes = dataset_manager.generate_prefix_data(train_current, min_prefix_length, max_prefix_length, gap)
    del train_current
            
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

    dt_preds_test = pd.DataFrame()
    dt_preds_val = pd.DataFrame()
            
    train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
            
    # initialize pipeline for sequence encoder and classifier
    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
    cls = ClassifierFactory.get_classifier(cls_method, cls_args, random_state, min_cases_for_training, overall_class_ratio)

    if cls_method == "svm" or cls_method == "logit":
        pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
    else:
        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

    # fit pipeline
    pipeline.fit(dt_train_prefixes, train_y)

    # predict for test
    preds = pipeline.predict_proba(dt_test_prefixes)
    dt_preds_test = pd.DataFrame({"predicted_proba": preds,
                                                                "actual": dataset_manager.get_label_numeric(dt_test_prefixes),
                                                                "prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                                                                "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})

    # predict for val
    preds = pipeline.predict_proba(dt_val_prefixes)
    dt_preds_val = pd.concat([dt_preds_val, pd.DataFrame({"predicted_proba": preds,
                                                            "actual": dataset_manager.get_label_numeric(dt_val_prefixes),
                                                            "prefix_nr": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                                                            "case_id": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})], axis=0)
        
    # write predictions
    dt_preds_test.to_csv(os.path.join(PREDS_DIR, "preds_test_%s_%s_%s.csv" % (dataset_name, method_name, cls_method)), sep=";", index=False)
    dt_preds_val.to_csv(os.path.join(PREDS_DIR, "preds_val_%s_%s_%s.csv" % (dataset_name, method_name, cls_method)), sep=";", index=False)
