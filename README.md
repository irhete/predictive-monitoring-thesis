# Predictive monitoring of business processes

Original datasets:
* Github data: https://github.com/riivo/github-issue-lifetime-prediction
* Public event logs: https://data.4tu.nl/repository/collection:event_logs_real

Preprocess datasets for predictive process monitoring:
* 

## Chapter 4: Benchmarking Existing Predictive Process Monitoring Techniques

1. Hyperparameter optimization
   - ``experiments_optimize_params.py``
2. Training and evaluating final models
   - ``experiments.py``
3. Execution times of final models
   - ``experiments_performance.py``

## Chapter 5: Predictive Monitoring with Structured and Unstructured Data

1. Hyperparameter optimization
1.1. experiments_optimize_params.py
1.2. experiments_optimize_params_with_unstructured_data.py
2. Training and evaluating final models
2.1. experiments_with_unstructured_data.py

## Chapter 6: Temporal Stability in Predictive Process Monitoring

1. Hyperparameter optimization
1.1. experiments_optimize_params.py
1.2. experiments_optimize_params_lstm.py
1.3. experiments_optimize_params_single_multirun.py
2. Training final models and writing predictions
2.1. experiments_stability.py

## Chapter 7: Alarm-Based Predictive Process Monitoring

1. Hyperparameter optimization
1.1. experiments_optimize_params.py
2. Training final models and writing predictions
2.1. experiments_write_predictions_alarms.py
3. Optimizing alarm thresholds
3.1. experiments_optimize_alarm_threshold.py
3.2. experiments_optimize_alarm_threshold_ccom.py
3.3. experiments_optimize_alarm_threshold_eff.py
4. Evaluating alarming thresholds
4.1. experiments_test_fixed_alarm_thresholds.py
4.2. experiments_test_optimal_alarm_threshold.py
4.3. experiments_test_optimal_alarm_threshold_ccom.py
4.4. experiments_test_optimal_alarm_threshold_eff.py



