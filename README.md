# Predictive monitoring of business processes

Original datasets:
* Github data: https://github.com/riivo/github-issue-lifetime-prediction
* Public event logs: https://data.4tu.nl/repository/collection:event_logs_real

Preprocessed datasets for predictive process monitoring:
* https://drive.google.com/file/d/1DDP7OKQhD8cno2tbSpLlIPZ-Mh5y-XUC/view?usp=sharing

## Chapter 4: Benchmarking Existing Predictive Process Monitoring Techniques

1. Hyperparameter optimization
   - ``experiments_optimize_params.py``
2. Training and evaluating final models
   - ``experiments.py``
3. Execution times of final models
   - ``experiments_performance.py``
4. Plots and tables
   - see https://github.com/irhete/predictive-monitoring-benchmark

## Chapter 5: Predictive Monitoring with Structured and Unstructured Data

1. Hyperparameter optimization
   - ``experiments_optimize_params.py``
   - ``experiments_optimize_params_with_unstructured_data.py``
2. Training and evaluating final models
   - ``experiments.py``
   - ``experiments_with_unstructured_data.py``
3. Execution times of final models
   - ``experiments_performance.py``
   - ``experiments_performance_with_unstructured_data.py``
4. Plots and tables
   - ``generate_latex_tables.ipynb``
   - ``plot_unstructured_results.R``

## Chapter 6: Temporal Stability in Predictive Process Monitoring

1. Hyperparameter optimization
   - ``experiments_optimize_params.py``
   - ``experiments_optimize_params_with_unstructured_data.py``
   - ``experiments_optimize_params_lstm.py``
   - ``experiments_optimize_params_single_multirun.py``
2. Training final models, calibrating, and writing predictions
   - ``experiments_write_predictions_stability.py``
   - ``experiments_write_predictions_stability_unstructured.py``
   - ``experiments_write_predictions_lstm.py``

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



