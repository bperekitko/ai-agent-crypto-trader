from typing import Type

import numpy as np
import pandas as pd

from model.evaluation.evaluate_binary_model import evaluate_binary_model
from model.model import Model

HOURS_IN_MONTH = 24 * 30


def validate_using_rolling_window(df: pd.DataFrame, model_cls: Type[Model], train_window_months: int) -> None:
    train_window = train_window_months * HOURS_IN_MONTH
    test_window = HOURS_IN_MONTH

    start = train_window
    all_probs = []
    all_true = []

    while start + test_window <= len(df):
        train_indices = range(start - train_window, start)
        test_indices = range(start, start + test_window)

        train_data = df.iloc[train_indices]
        test_data = df.iloc[test_indices]
        model = model_cls()
        model.train(train_data)
        probabilities, true_y = model.predict(test_data)

        all_probs.append(probabilities)
        all_true.append(true_y)
        start += test_window

    model = model_cls()
    model.params['rolling_window'] = train_window_months
    evaluate_binary_model(np.vstack(all_probs), np.concatenate(all_true, axis=0), model)
