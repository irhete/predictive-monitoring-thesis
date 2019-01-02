"""This script trains a predictive model for outcome-oriented predictive process monitoring and writes the predictions to a file.

Usage:
    experiments_write_predictions_stability.py <dataset> <method> <classifier>

Example:
    experiments_write_predictions_stability.py bpic2012_cancelled single_laststate xgboost
  
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

from sklearn.calibration import CalibratedClassifierCV


PARAMS_DIR = "val_results_unstructured"
RESULTS_DIR = "results"
DETAILED_RESULTS_DIR = "%s_detailed" % RESULTS_DIR

dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]

bucket_method, cls_encoding = method_name.split("_")

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
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
if not os.path.exists(DETAILED_RESULTS_DIR):
    os.makedirs(DETAILED_RESULTS_DIR)
    
for dataset_name in datasets:
    
    detailed_results = pd.DataFrame()
    detailed_results_val = pd.DataFrame()
    
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
    train, test = dataset_manager.split_data_strict(data, train_ratio)
    del data
    if "calibrate" in cls_method:
        train, val = dataset_manager.split_val(train, val_ratio)
    overall_class_ratio = dataset_manager.get_class_ratio(train)
    
    if method_name == "prefix_index":
        nr_eventss = range(min_prefix_length, max_prefix_length+1)
    else:
        nr_eventss = [None]
        
    print("Split data")
    
    for nr_events in nr_eventss:
        if "single" in method_name:
            optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method.replace("_calibrated", ""), dataset_name, method_name, text_method_enc))
        else:
            optimal_params_filename = os.path.join(PARAMS_DIR, "optimal_params_%s_%s_%s_%s_%s.pickle" % (cls_method.replace("_calibrated", ""), dataset_name, method_name, nr_events, text_method_enc))
            
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
          
        text_transformer = EncoderFactory.get_encoder(text_method, text_transformer_args=text_transformer_args)
        dt_train_text = text_transformer.fit_transform(train[dataset_manager.static_text_cols+dataset_manager.dynamic_text_cols], 
                                                       train[dataset_manager.label_col])
        
        static_text_cols = []
        dynamic_text_cols = []
        for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
            dt_train_text = text_transformer.transform(train[[col]], train[dataset_manager.label_col])
            current_text_cols = ["%s_%s" % (col, text_col) for text_col in dt_train_text.columns]
            dt_train_text.columns = current_text_cols
            train_current = pd.concat([train.drop(col, axis=1), dt_train_text], axis=1, sort=False)
            del dt_train_text
            if "single" in method_name:
                del train
                
            dt_test_text = text_transformer.transform(test[[col]])
            dt_test_text.columns = current_text_cols
            test_current = pd.concat([test.drop(col, axis=1), dt_test_text], axis=1, sort=False)
            del dt_test_text
            if "single" in method_name:
                del test
                
            if "calibrate" in cls_method:
                dt_val_text = text_transformer.transform(val[[col]])
                dt_val_text.columns = current_text_cols
                val_current = pd.concat([val.drop(col, axis=1), dt_val_text], axis=1, sort=False)
                del dt_val_text
                if "single" in method_name:
                    del val
                    
            if col in dataset_manager.static_text_cols:
                static_text_cols.extend(current_text_cols)
            else:
                dynamic_text_cols.extend(current_text_cols)

        print("Transformed text")
                
        # generate prefixes
        if "single" in method_name:
            dt_train_bucket = dataset_manager.generate_prefix_data(train_current, min_prefix_length, max_prefix_length)
            del train_current
            
            if "calibrate" in cls_method:
                dt_val_bucket = dataset_manager.generate_prefix_data(val_current, min_prefix_length, max_prefix_length)
                del val_current
        else:
            dt_train_bucket = dataset_manager.generate_prefix_data(train_current, nr_events, nr_events)
            dt_test_bucket = dataset_manager.generate_prefix_data(test_current, nr_events, nr_events)
            if "calibrate" in cls_method:
                dt_val_bucket = dataset_manager.generate_prefix_data(val_current, nr_events, nr_events)
        
        print("Generated prefixes")
        
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

        X_train = feature_combiner.fit_transform(dt_train_bucket)
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)
        # fit classifier and calibrate
        cls = ClassifierFactory.get_classifier(cls_method.replace("_calibrated", ""), cls_args, random_state, min_cases_for_training, overall_class_ratio, binary=(False if "calibrate" in cls_method else True))
        cls.fit(X_train, train_y)
        del X_train, train_y
        
        print("Trained model")

        if "calibrate" in cls_method:
            X_val = feature_combiner.transform(dt_val_bucket)
            y_val = dataset_manager.get_label_numeric(dt_val_bucket)
            
            cls = CalibratedClassifierCV(cls, cv="prefit", method='sigmoid')
            cls.fit(X_val, np.array(y_val))

        print("Calibrated model")
            
        if "single" in method_name:
            dt_test_bucket = dataset_manager.generate_prefix_data(test_current, min_prefix_length, max_prefix_length)
        del test_current
        X_test = feature_combiner.transform(dt_test_bucket)
        # predict 
        preds = cls.predict_proba(X_test)
        del X_test
        if "calibrate" in cls_method:
            preds = preds[:,1]
        
        print("Predicted")
        
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
        case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
        del dt_test_bucket
        
        current_results = pd.DataFrame({"dataset": dataset_name, "cls": cls_method, "params": method_name, "nr_events": nr_events, "predicted": preds, "actual": test_y, "case_id": case_ids})
        detailed_results = pd.concat([detailed_results, current_results], axis=0, sort=False)
        
        if "calibrate" in cls_method:
            preds = cls.predict_proba(X_val)[:,1]
            case_ids = list(dt_val_bucket.groupby(dataset_manager.case_id_col).first().index)
            current_results = pd.DataFrame({"dataset": dataset_name, "cls": cls_method, "params": method_name, "nr_events": nr_events, "predicted": preds, "actual": y_val, "case_id": case_ids})
            detailed_results_val = pd.concat([detailed_results_val, current_results], axis=0, sort=False)

        print("Predicted for val")
            
    # write results
    #outfile = os.path.join(RESULTS_DIR, "results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    detailed_results_file = os.path.join(DETAILED_RESULTS_DIR, "detailed_results_test_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
    detailed_results.to_csv(detailed_results_file, sep=";", index=False)
    if "calibrate" in cls_method:
        detailed_results_file = os.path.join(DETAILED_RESULTS_DIR, "detailed_results_val_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
        detailed_results_val.to_csv(detailed_results_file, sep=";", index=False)
