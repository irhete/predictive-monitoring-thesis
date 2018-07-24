from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ClassifierWrapper import ClassifierWrapper
from sklearn.calibration import CalibratedClassifierCV


def get_classifier(method, current_args, random_state=None, min_cases_for_training=30, hardcoded_prediction=0.5):

    if method == "rf":
        return ClassifierWrapper(RandomForestClassifier(n_estimators=current_args['n_estimators'],
                                                       max_features=current_args['max_features'],
                                                       random_state=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)

    elif method == "xgboost":
        return ClassifierWrapper(xgb.XGBClassifier(objective='binary:logistic',
                                                  n_estimators=current_args['n_estimators'],
                                                  learning_rate= current_args['learning_rate'],
                                                  subsample=current_args['subsample'],
                                                  max_depth=int(current_args['max_depth']),
                                                  colsample_bytree=current_args['colsample_bytree'],
                                                  min_child_weight=int(current_args['min_child_weight']),
                                                  seed=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)

    elif method == "logit":
        return ClassifierWrapper(LogisticRegression(C=2**current_args['C'],
                                                   random_state=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)

    elif method == "svm":
        return ClassifierWrapper(SVC(C=2**current_args['C'],
                                    gamma=2**current_args['gamma'],
                                    random_state=random_state), 
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)
    
    
    elif method == "gbm":
        return ClassifierWrapper(
            cls=GradientBoostingClassifier(n_estimators=current_args['n_estimators'],
                                           max_features=current_args['max_features'],
                                           learning_rate=current_args['learning_rate'],
                                           random_state=random_state),
            method=method,
            min_cases_for_training=min_cases_for_training,
            hardcoded_prediction=hardcoded_prediction)
    
    elif method == "dt":
        return ClassifierWrapper(
            cls=DecisionTreeClassifier(random_state=random_state),
            method=method,
            min_cases_for_training=min_cases_for_training,
            hardcoded_prediction=hardcoded_prediction)
    
    elif method == "calibrated_rf_sigmoid":
        clf = RandomForestClassifier(n_estimators=current_args['n_estimators'],
                                     max_features=current_args['max_features'],
                                     random_state=random_state)
        return ClassifierWrapper(cls=CalibratedClassifierCV(clf, cv=5, method='sigmoid'),
                                 method=method,
                                 min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)
    
    elif method == "calibrated_rf_isotonic":
        clf = RandomForestClassifier(n_estimators=current_args['n_estimators'],
                                     max_features=current_args['max_features'],
                                     random_state=random_state)
        return ClassifierWrapper(cls=CalibratedClassifierCV(clf, cv=5, method='isotonic'),
                                 method=method,
                                 min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction)
               
    else:
        print("Invalid classifier type")
        return None