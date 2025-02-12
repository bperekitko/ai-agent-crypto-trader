import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
    brier_score_loss,
    matthews_corrcoef
)

from model.evaluation import current_dir_path
from model.model import Model
from utils.add_to_excel import append_df_to_excel
from utils.log import get_logger

__CONFIDENCE_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
_LOG = get_logger("Binary Model Evaluation")


def evaluate_binary_model(y_pred_prob, y_true, model: Model):
    _LOG.info(f'Evaluating model {model.name()}')
    flattened_probs = y_pred_prob.flatten()
    for confidence in __CONFIDENCE_LEVELS:
        __evaluate_with_confidence(model, flattened_probs, y_true, confidence)
    _LOG.info(f'Saved evaluation results to file: {model.name()}_{model.version()}.xlsx')

def evaluate_simple_stats(y_pred, y_true, model_name):
    _LOG.info(f'Evaluating model {model_name}')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    results = {
        'Model': f'{model_name}',
        'Predictions: ': np.sum(y_pred == 1),
        'True: ': np.sum(y_true == 1),
        'Total samples': len(y_pred),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'F1-Score': round(f1, 3),
        'Cohen Kappa': round(cohen, 3),
        'MCC': round(mcc, 3),
        'Total Accuracy': round(acc, 3),
    }
    results_df = pd.DataFrame([results])
    append_df_to_excel(results_df, current_dir_path(f'{model_name}.xlsx'))
    _LOG.info(f'Saved evaluation results to file: {model_name}.xlsx')

def __evaluate_with_confidence(model: Model, probabilities, y_test, confidence):
    y_pred = (probabilities > confidence).astype(int).astype(float)
    if np.sum(y_pred == 1) == 0:
        _LOG.warn(f'No predictions with confidence level: {confidence}')
        return

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cohen = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    ll = log_loss(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    brier = brier_score_loss(y_test, probabilities)

    results = {
        'Model': f'{model.name()}_{model.version()}',
        'Confidence threshold': confidence,
        'Predictions: ': np.sum(y_pred == 1),
        'True: ': np.sum(y_test == 1),
        'Total samples': len(y_test),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'F1-Score': round(f1, 3),
        'Cohen Kappa': round(cohen, 3),
        'Log Loss': round(ll, 3),
        'MCC': round(mcc, 3),
        'Total Accuracy': round(acc, 3),
        'AUC': round(auc, 3),
        'Brier Score': round(brier, 3),
        **model.params
    }
    results_df = pd.DataFrame([results])
    append_df_to_excel(results_df, current_dir_path(f'{model.name()}_{model.version()}.xlsx'))
