import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score

from model.evaluation import current_dir_path
from model.features.target import TargetLabel
from model.model import Model
from utils.add_to_excel import append_df_to_excel
from utils.log import get_logger

__SIGNAL_FREQUENCY_THRESHOLD = 0.05
__CONFIDENCE_LEVELS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6]
_LOG = get_logger("Model Evaluation")


def evaluate_for_highs_and_lows(probabilities, model: Model, input_data: pd.DataFrame, threshold_up_provider, threshold_down_provider):
    df = input_data.copy()

    y_pred = np.argmax(probabilities, axis=1)
    dropna = False
    if len(y_pred) < len(df):
        na_padding = [np.nan] * (len(df) - len(y_pred))
        y_pred = na_padding + list(y_pred)
        dropna = True

    df['pred'] = y_pred
    df['high_percent_change'] = ((df['high'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    df['low_percent_change'] = ((df['low'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
    df['threshold_up'] = threshold_up_provider(df)
    df['threshold_down'] = threshold_down_provider(df)

    def adjust_target(df_row):
        if df_row['pred'] == TargetLabel.UP.value:
            if df_row['high_percent_change'] > df_row['threshold_up']:
                return TargetLabel.UP.value
            else:
                return df_row['target']
        elif df_row['pred'] == TargetLabel.DOWN.value:
            if df_row['low_percent_change'] < df_row['threshold_down']:
                return TargetLabel.DOWN.value
            else:
                return df_row['target']
        else:
            if df_row['high_percent_change'] > df_row['threshold_up']:
                return TargetLabel.UP.value
            elif df_row['low_percent_change'] < df_row['threshold_down']:
                return TargetLabel.DOWN.value
            else:
                return TargetLabel.NEUTRAL.value

    if dropna:
        df.dropna(inplace=True)

    df['adjusted_target'] = df.apply(adjust_target, axis=1)
    model.params['adjusted_target'] = True
    evaluate(probabilities, df['adjusted_target'], model)


def evaluate(probabilities, y_test, model: Model):
    _LOG.info(f'Evaluating model {model.name()}')
    original_count = len(probabilities)

    for confidence in __CONFIDENCE_LEVELS:
        confident_predictions_filter = np.max(probabilities, axis=1) > confidence
        confident_probabilities = probabilities[confident_predictions_filter]
        confident_y_true = y_test[confident_predictions_filter]
        if len(confident_y_true > 0):
            __evaluate_with_confidence(model, confident_probabilities, confident_y_true, confidence, original_count)
        else:
            _LOG.warn(f'No samples with confidence level: {confidence}')
    _LOG.info(f'Saved evaluation results to file: {model.name()}_{model.version()}.xlsx')


def __evaluate_with_confidence(model: Model, probabilities, y_test, confidence, original_pred_count):
    log_loss_value = log_loss(y_test, probabilities, labels=[0, 1, 2])
    roc_auc_value = roc_auc_score(y_test, probabilities, multi_class='ovr', labels=[0, 1, 2])
    brier_scores = [brier_score_loss((y_test == i).astype(int), probabilities[:, i]) for i in
                    range(probabilities.shape[1])]
    y_pred = np.argmax(probabilities, axis=1)

    unique_values, counts = np.unique(y_pred, return_counts=True)
    count_dict = {0: 0, 1: 0, 2: 0}
    count_dict.update(dict(zip(unique_values, counts)))

    kappa_score = cohen_kappa_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=np.nan)
    recall = recall_score(y_test, y_pred, labels=[0, 1, 2], average=None, zero_division=np.nan)
    f1 = f1_score(y_test, y_pred, labels=[0, 1, 2], average=None)
    mcc = matthews_corrcoef(y_test, y_pred)

    signals_frequency = (count_dict[TargetLabel.UP.value] + count_dict[TargetLabel.DOWN.value]) / original_pred_count
    if signals_frequency <= __SIGNAL_FREQUENCY_THRESHOLD:
        _LOG.warn(f'Not enough signals frequency to create meaningful stats: {round(signals_frequency, 2)}, ignoring confidence: {confidence}')
        return

    down_precision = 0 if np.isnan(precision[0]) else precision[0]
    up_precision = 0 if np.isnan(precision[1]) else precision[1]
    neutral_precision = 0 if np.isnan(precision[2]) else precision[2]

    signal_avg_precision = down_precision if up_precision == 0 else up_precision if down_precision == 0 else round((down_precision + up_precision) / 2, 3)
    signal_avg_recall = recall[0] if recall[1] == 0 else recall[1] if recall[0] == 0 else round((recall[0] + recall[1]) / 2, 3)
    signal_avg_f1 = f1[0] if f1[1] == 0 else f1[1] if f1[0] == 0 else round((f1[0] + f1[1]) / 2, 3)

    results = {
        'Model': f'{model.name()}_{model.version()}',
        'Confidence threshold': confidence,
        'Trade Signals Frequency': signals_frequency,
        'SignalsUP': count_dict[TargetLabel.UP.value],
        'SignalsDOWN': count_dict[TargetLabel.DOWN.value],
        'NoTrade': count_dict[TargetLabel.NEUTRAL.value],
        'Avg Signal Precision': round(signal_avg_precision, 3),
        'Avg Signal Recall': round(signal_avg_recall, 3),
        'Avg Signal F1': round(signal_avg_f1, 3),
        f'{TargetLabel(0).name}_precision': round(down_precision, 3),
        f'{TargetLabel(1).name}_precision': round(up_precision, 3),
        f'{TargetLabel(2).name}_precision': round(neutral_precision, 3),

        f'{TargetLabel(0).name}_recall': round(recall[0], 3),
        f'{TargetLabel(1).name}_recall': round(recall[1], 3),
        f'{TargetLabel(2).name}_recall': round(recall[2], 3),

        f'{TargetLabel(0).name}_F1': round(f1[0], 3),
        f'{TargetLabel(1).name}_F1': round(f1[1], 3),
        f'{TargetLabel(2).name}_F1': round(f1[2], 3),
        'Cohen Kappa': round(kappa_score, 3),
        'Log Loss': round(log_loss_value, 3),
        'MCC': round(mcc, 3),
        'Total Accuracy': round(accuracy, 3),
        'ROC AUC': round(roc_auc_value, 3),
        'Brier Score (DOWN)': round(brier_scores[0], 3),
        'Brier Score (UP)': round(brier_scores[1], 3),
        'Brier Score (NEUTRAL)': round(brier_scores[2], 3),
        **model.params
    }
    results_df = pd.DataFrame([results])
    append_df_to_excel(results_df, current_dir_path(f'{model.name()}_{model.version()}.xlsx'))
