datasets = ['bpic2011_f1',
             'bpic2011_f2',
             'bpic2011_f3',
             'bpic2011_f4',
             'bpic2015_1_f2',
             'bpic2015_2_f2',
             'bpic2015_3_f2',
             'bpic2015_4_f2',
             'bpic2015_5_f2',
             'production',
             'insurance_activity',
             'insurance_followup',
             'sepsis_cases_1',
             'sepsis_cases_2',
             'sepsis_cases_4',
             'bpic2012_accepted',
             'bpic2012_declined',
             'bpic2012_cancelled',
             'bpic2017_accepted',
             'bpic2017_refused',
             'bpic2017_cancelled',
             'traffic_fines_1',
             'hospital_billing_2',
             'hospital_billing_3',
           'dc',
           'crm2',
           'github',
           'unemployment']


from DatasetManager import DatasetManager

train_ratio = 0.8

test_cases_dict = {}

for dataset_name in datasets:
    
    test_cases_dict[dataset_name] = {}

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
    prefix_lengths = list(range(min_prefix_length, max_prefix_length + 1))

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

    test_case_lengths = test.groupby(dataset_manager.case_id_col).size()
    
    for nr_events in prefix_lengths:
        test_cases_dict[dataset_name][nr_events] = len(test_case_lengths.index[test_case_lengths >= nr_events])
    
    n_test_cases = [str(len(test_case_lengths.index[test_case_lengths >= nr_events])) for nr_events in prefix_lengths]
    print("%s: %s"%(dataset_name, ", ".join(n_test_cases)))
    
    
import pickle

with open("n_test_cases.pickle", "wb") as fout:
    pickle.dump(test_cases_dict, fout)