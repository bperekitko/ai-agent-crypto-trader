import numpy as np
import pandas
from dateutil.relativedelta import relativedelta

from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate
from model.lstm.lstm import Lstm


def time_series_cross_validate(df: pandas.DataFrame):
    train_start = df.head(1)[DataColumns.DATE_OPEN].values[0].astype('datetime64[us]').astype('O')
    last_candle = df.tail(1)[DataColumns.DATE_OPEN].values[0].astype('datetime64[us]').astype('O')
    res = (last_candle.year - train_start.year) * 12 + (last_candle.month - train_start.month)

    data_sets = __prepare_data_sets(df, res, train_start)

    total_probabilities = None
    total_y_true = None
    model = None
    for one_set in data_sets:
        model = Lstm()
        train_set, test_set = one_set
        model.train(train_set)
        probs, y_true = model.test(test_set)
        total_probabilities = probs if total_probabilities is None else np.concatenate((total_probabilities, probs), axis=0)
        total_y_true = y_true if total_y_true is None else np.append(total_y_true, y_true)

    model.params['tested_on'] = f'FROM_{3}_TOTAL'
    model.params['trained_on'] = f'FROM_{3}_TOTAL'
    evaluate(total_probabilities, total_y_true, model)


def __filtered_by(df, start, end):
    filtered_by_dates = (df[DataColumns.DATE_CLOSE] >= start) & (df[DataColumns.DATE_CLOSE] <= end)
    return df.loc[filtered_by_dates]


def __prepare_data_sets(df, res, train_start):
    data_sets = []
    for i in range(3, res):
        train_end = train_start + relativedelta(months=i)
        test_start = train_end
        test_end = test_start + relativedelta(months=1)

        train_data = __filtered_by(df, train_start, train_end)
        test_data = __filtered_by(df, test_start, test_end)
        set_tuple = (train_data, test_data)
        data_sets.append(set_tuple)
    return data_sets
