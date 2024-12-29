import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from model.softmax_regression.evaluation import current_dir_path
from utils.add_to_excel import append_df_to_excel


def evaluate(probabilities, y_test, model_params, model_name, model_version):

    log_loss_value = log_loss(y_test, probabilities)
    roc_auc_value = roc_auc_score(y_test, probabilities, multi_class='ovr')
    y_true= np.array(y_test)
    brier_scores = [brier_score_loss((y_true == i).astype(int), probabilities[:, i]) for i in
                    range(probabilities.shape[1])]
    y_pred = np.argmax(probabilities, axis=1)
    kappa_score = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    results = {
        'Model': f'{model_name}_{model_version}',
        'Accuracy': accuracy,
        'Log Loss': log_loss_value,
        'ROC AUC': roc_auc_value,
        'Brier Score (0)': brier_scores[0],
        'Brier Score (1)': brier_scores[1],
        'Brier Score (2)': brier_scores[2],
        'Cohen Kappa': kappa_score,
        'ModelParams': json.dumps(model_params)
    }

    results_df = pd.DataFrame([results])
    append_df_to_excel(results_df, current_dir_path('softmax_linear_regression.xlsx'))
