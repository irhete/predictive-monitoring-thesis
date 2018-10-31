import glob
import pandas as pd
import os
import numpy as np
import pickle
import sys
import string
from collections import Counter

import dataset_confs
from DatasetManager import DatasetManager

home_dir = ""

datasets = ["dc", "crm2", "github"]

translator = str.maketrans('','',string.punctuation)

general_table = []
text_table = []

for dataset_name in datasets:
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    
    """
        data["%s_cleaned" % col] = cleaned_texts

    if dataset_name != "dc":
        data.to_csv(dataset_confs.filename[dataset_name], sep=";", index=False)
   """
        
    """
    data["text_combined"] = data[dataset_manager.text_cols[0]]
    for col in dataset_manager.text_cols[1:]:
        data["text_combined"] += " " + data[col]
    data.to_csv(dataset_confs.filename[dataset_name], sep=";", index=False)
    
    
    """
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
            
    data = dataset_manager.read_dataset()
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
    
    vocabulary = set()
    doc_lengths_dynamic = []
    doc_lengths_static = []
    n_nonempty = 0
    for col in dataset_manager.static_text_cols + dataset_manager.dynamic_text_cols:
        for text in data[col]:
            parts = text.split()
            if len(parts) > 0:
                if col in dataset_manager.dynamic_text_cols:
                    n_nonempty += 1
                    doc_lengths_dynamic.append(len(parts))
                else:
                    doc_lengths_static.append(len(parts))
            vocabulary.update(parts)

    doc_lengths_dynamic = pd.Series(doc_lengths_dynamic)
    doc_lengths_static = pd.Series(doc_lengths_static)
    
    dataset_name = dataset_name.replace("_", "\\_").replace("bpic2011\_f", "bpic2011\_").replace("\_f2", "").replace("activity", "1").replace("followup", "2").replace("cases\_", "").replace("billing\_", "").replace("accepted", "1").replace("declined", "2").replace("refused", "2").replace("cancelled", "3").replace("\_fines\_1", "").replace("sepsis\_4", "sepsis\_3").replace("hospital\_2", "hospital\_1").replace("hospital\_3", "hospital\_2")
    
    general_table.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (dataset_name, len(data[case_id_col].unique()), sizes.min(), sizes.quantile(0.50), sizes.max(), max_prefix_length, n_trace_variants, round(class_freqs[pos_label] / len(data[case_id_col].unique()), 2), len(data[activity_col].unique()), len(static_cat_cols) + len(static_num_cols), len(dynamic_cat_cols) + len(dynamic_num_cols), n_static_cat_levels, n_dynamic_cat_levels))
    
    text_table.append("%s & %s & %s & %s & %s \\\\" % (dataset_name, len(vocabulary), round(doc_lengths_static.mean(), 2), round(doc_lengths_dynamic.mean(), 2), round(n_nonempty / len(data), 2)))
    
    """
    record = (dataset_name, len(data[case_id_col].unique()), sizes.min(), sizes.quantile(0.50), sizes.max(), max_prefix_length, n_trace_variants, round(class_freqs[pos_label] / len(data[case_id_col].unique()), 2), len(data[activity_col].unique()), len(static_cat_cols) + len(static_num_cols), len(dynamic_cat_cols) + len(dynamic_num_cols),
                                                                      n_static_cat_levels, n_dynamic_cat_levels)
    dt_stats.append(record)
    """
    
    """
    train, test = dataset_manager.split_data_strict(data, 0.8, split="temporal")
    print("Train temporal: ", train.groupby(case_id_col).first()[label_col].value_counts() / len(train.groupby(case_id_col)))
    print("Test temporal: ", test.groupby(case_id_col).first()[label_col].value_counts() / len(test.groupby(case_id_col)))
    
    train, test = dataset_manager.split_data(data, 0.8, split="random", seed=22)
    print("Train random: ", train.groupby(case_id_col).first()[label_col].value_counts() / len(train.groupby(case_id_col)))
    print("Test random: ", test.groupby(case_id_col).first()[label_col].value_counts() / len(test.groupby(case_id_col)))
    """
    
    #print(col, len(word_counter), len(word_counter_cleaned), doc_lengths.min(), doc_lengths.median(), doc_lengths.mean(), doc_lengths.max())

    
print("\\toprule  &  & min  & med  & max  & trunc  & \\# variants & pos class  & \\# event & \\# static  & \\# dynamic  & \\# static  & \\# dynamic \\\\ ")
print(" dataset & \\# traces &  length &  length &  length &  length & (after trunc) & ratio & classes &  attr-s & attr-s &  cat levels & cat levels \\\\ \\midrule")
for line in general_table:
    print(line)
print("\\bottomrule")

    
print("\\toprule  &  \\# lemmas & avg. length (static) & avg. length (dynamic) & \\% events with text \\\\ \\midrule")
for line in text_table:
    print(line)
print("\\bottomrule")