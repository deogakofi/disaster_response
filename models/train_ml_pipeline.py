import ml_pipeline as ml
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import make_scorer, precision_score, classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

X, Y = ml.load_data('emergency', 'disaster')

def run_rf_model():
    model_rf = ml.build_model(RandomForestClassifier())
    X_train_rf, X_test_rf, y_train_rf, y_test_rf, y_pred_rf, model_rf_fit = ml.train_model(X, Y, model_rf)
    results_rf = ml.get_results(y_test_rf, y_pred_rf)
    return results_rf, model_rf, model_rf_fit

results_rf, model_rf, model_rf_fit = run_rf_model()



def save_rf_model():
    ml.save_model('model_rf', model_rf)
    ml.save_model('model_rf_fit', model_rf_fit)

save_rf_model()

def run_ada_model():
    model_ada = ml.build_model(AdaBoostClassifier())
    X_train_ada, X_test_ada, y_train_ada, y_test_ada, y_pred_ada, model_ada_fit = ml.train_model(X, Y, model_ada)
    results_ada = ml.get_results(y_test_ada, y_pred_ada)
    return results_ada, model_ada, model_ada_fit

results_ada, model_ada, model_ada_fit = run_ada_model()

def save_ada_model():
    ml.save_clf_results('results_ada', results_ada)
    ml.save_model('model_ada', model_ada)
    ml.save_model('model_ada_fit', model_ada_fit)

save_ada_model()
