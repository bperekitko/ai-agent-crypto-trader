import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score

from model.evaluation import current_dir_path
from utils.add_to_excel import append_df_to_excel
from utils.log import Logger

__CONFIDENCE_LEVELS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6]


def evaluate(probabilities, y_test, model_params, model_name):
    logger = Logger(model_name)
    logger.info(f'Evaluating model {model_name}')
    original_count = len(probabilities)

    for confidence in __CONFIDENCE_LEVELS:
        confident_predictions_filter = np.max(probabilities, axis=1) > confidence
        confident_probabilities = probabilities[confident_predictions_filter]
        confident_y_true = y_test[confident_predictions_filter]
        if len(confident_y_true > 0):
            __evaluate_with_confidence(model_name, model_params, confident_probabilities, confident_y_true, confidence, original_count)
        else:
            logger.warn(f'No samples with confidence level: {confidence}')
    logger.info(f'Saved evaluation results to file: {model_name}.xlsx')


def __evaluate_with_confidence(model_name, model_params, probabilities, y_test, confidence, original_pred_count):
    log_loss_value = log_loss(y_test, probabilities, labels=[0, 1, 2])
    roc_auc_value = roc_auc_score(y_test, probabilities, multi_class='ovr', labels=[0, 1, 2])
    brier_scores = [brier_score_loss((y_test == i).astype(int), probabilities[:, i]) for i in
                    range(probabilities.shape[1])]
    y_pred = np.argmax(probabilities, axis=1)
    kappa_score = cohen_kappa_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=np.nan)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=np.nan)
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    results = {
        'Model': f'{model_name}',
        'Confidence threshold': confidence,
        'Confident predictions count': f'{len(y_test)} of {original_pred_count}',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'MCC': mcc,
        'Cohen Kappa': kappa_score,
        'Log Loss': log_loss_value,
        'ROC AUC': roc_auc_value,
        'Brier Score (DOWN)': brier_scores[0],  # {'DOWN': 0, 'UP': 1, 'NEUTRAL': 2}
        'Brier Score (UP)': brier_scores[1],
        'Brier Score (NEUTRAL)': brier_scores[2],
        'ModelParams': json.dumps(model_params)
    }
    results_df = pd.DataFrame([results])
    append_df_to_excel(results_df, current_dir_path(f'{model_name}.xlsx'))
