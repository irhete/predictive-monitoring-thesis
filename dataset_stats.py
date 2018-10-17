import glob
import pandas as pd
import os
import numpy as np
import pickle
import sys

import dataset_confs
from DatasetManager import DatasetManager

home_dir = ""

datasets = ["dc", "crm2", "github"]

dt_stats = []

print("\\toprule  &  & min  & med  & max  & trunc  & \\# variants & pos class  & \\# event & \\# static  & \\# dynamic  & \\# static  & \\# dynamic \\\\ ")
print(" dataset & \\# traces &  length &  length &  length &  length & (after trunc) & ratio & classes &  attr-s & attr-s &  cat levels & cat levels\\\\ \\midrule")
for dataset_name in datasets:
    dataset_manager = DatasetManager(dataset_name)
    
    case_id_col = dataset_confs.case_id_col[dataset_name]
    activity_col = dataset_confs.activity_col[dataset_name]
    timestamp_col = dataset_confs.timestamp_col[dataset_name]
    label_col = dataset_confs.label_col[dataset_name]
    pos_label = dataset_confs.pos_label[dataset_name]

    dynamic_cat_cols = dataset_confs.dynamic_cat_cols[dataset_name]
    static_cat_cols = dataset_confs.static_cat_cols[dataset_name]
    dynamic_num_cols = dataset_confs.dynamic_num_cols[dataset_name]
    static_num_cols = dataset_confs.static_num_cols[dataset_name]

    data_filepath = os.path.join(home_dir, dataset_confs.filename[dataset_name])

    # specify data types
    dtypes = {col:"object" for col in dynamic_cat_cols+static_cat_cols+[case_id_col, label_col, timestamp_col]}
    for col in dynamic_num_cols + static_num_cols:
        dtypes[col] = "float"
            
    data = pd.read_csv(data_filepath, sep=";", dtype=dtypes)
    sizes = data.groupby(case_id_col).size()

    class_freqs = data.groupby(case_id_col).first()[label_col].value_counts()
    
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    
    data[case_id_col] = data[case_id_col].astype(str)
    data[activity_col] = data[activity_col].astype(str)
    n_trace_variants = len(data.sort_values(timestamp_col, kind="mergesort").groupby(case_id_col).head(max_prefix_length).groupby(case_id_col)[activity_col].apply(lambda x: "__".join(list(x))).unique())
    
    n_static_cat_levels = 0
    n_dynamic_cat_levels = 0
    for col in dynamic_cat_cols:
        n_dynamic_cat_levels += len(data[col].unique())
    for col in static_cat_cols:
        n_static_cat_levels += len(data[col].unique())
    
    dataset_name = dataset_name.replace("_", "\\_").replace("bpic2011\_f", "bpic2011\_").replace("\_f2", "").replace("activity", "1").replace("followup", "2").replace("cases\_", "").replace("billing\_", "").replace("accepted", "1").replace("declined", "2").replace("refused", "2").replace("cancelled", "3").replace("\_fines\_1", "").replace("sepsis\_4", "sepsis\_3").replace("hospital\_2", "hospital\_1").replace("hospital\_3", "hospital\_2")
    print("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\"%(dataset_name, len(data[case_id_col].unique()), sizes.min(), sizes.quantile(0.50), sizes.max(), max_prefix_length, n_trace_variants, round(class_freqs[pos_label] / len(data[case_id_col].unique()), 2), len(data[activity_col].unique()), len(static_cat_cols) + len(static_num_cols), len(dynamic_cat_cols) + len(dynamic_num_cols),
                                                                      n_static_cat_levels, n_dynamic_cat_levels))
    
    record = (dataset_name, len(data[case_id_col].unique()), sizes.min(), sizes.quantile(0.50), sizes.max(), max_prefix_length, n_trace_variants, round(class_freqs[pos_label] / len(data[case_id_col].unique()), 2), len(data[activity_col].unique()), len(static_cat_cols) + len(static_num_cols), len(dynamic_cat_cols) + len(dynamic_num_cols),
                                                                      n_static_cat_levels, n_dynamic_cat_levels)
    dt_stats.append(record)
    
print("\\bottomrule")