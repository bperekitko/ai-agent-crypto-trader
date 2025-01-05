import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from data.raw_data import get_data
from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate
from model.softmax_regression.softmax import SoftmaxRegression


def main():
    for i in range(9, 12):
        evaluate_for_train_months(i, SoftmaxRegression)

    # train_data = get_data(datetime(2024, 1, 1), datetime(2024, 11, 1))
    # test_data = get_data(datetime(2024, 11, 1), datetime(2025, 3, 2))
    #
    # for i in range(5,25):
    #     model = SoftmaxRegression()
    #     model.set_window(i)
    #     model.train(train_data)
    #     model.test(test_data)

    # model = Lstm()
    #
    # model.train(train_data)
    # predictions = model.predict(test_data)
    # result = pd.DataFrame(predictions, columns=['DOWN', 'UP', 'NEUTRAL'])
    # result[DataColumns.TARGET] = model.prepare_for_predict(test_data)[DataColumns.TARGET].values
    #
    # test_data = get_data(datetime(2024, 3, 2), datetime(2024, 3, 3))
    # probs = model.predict(test_data)
    # result2 = pd.DataFrame(predictions, columns=['DOWN', 'UP', 'NEUTRAL'])
    # result2[DataColumns.TARGET] = model.prepare_for_predict(test_data)[DataColumns.TARGET].values
    #
    # print(f'Len1: {len(result)}, len2: {len(result2)}')
    #
    # print(pd.concat([result, result2], ignore_index=True))

    # evaluate(predictions, test_data[DataColumns.TARGET].values, {}, 'softmax_linear_regression')


def evaluate_for_train_months(train_months, model_func):
    # for i in range(5, 25):
    # model = SoftmaxRegression()
    #     model.set_window(i)
    #     model.train(train_data)
    #     model.test(test_data)
    total = pd.DataFrame()
    train_start = datetime(2024, 1, 1)
    while train_start + relativedelta(months=train_months) <= datetime(2024, 12, 1):
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=1)

        train_data = get_data(train_start, train_end)
        test_data = get_data(test_start, test_end)

        model = model_func()
        # model.set_window(i)
        model.train(train_data)
        predictions = model.predict(test_data)

        result = pd.DataFrame(predictions, columns=['DOWN', 'UP', 'NEUTRAL'])
        result[DataColumns.TARGET] = model.prepare_for_predict(test_data)[DataColumns.TARGET].values
        total = pd.concat([total, result], ignore_index=True)

        train_start = train_start + relativedelta(months=1)

    test_y = total[DataColumns.TARGET]
    evaluate(total[['DOWN', 'UP', 'NEUTRAL']].values, np.array(test_y), {"train_size": train_months, 'percentiles': '50:50'}, SoftmaxRegression().name())


if __name__ == "__main__":
    main()
