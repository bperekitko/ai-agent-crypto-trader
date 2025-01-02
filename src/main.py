from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from data.raw_data import get_data
from data.raw_data_columns import DataColumns
from model.softmax_regression.evaluation.evaluate import evaluate
from model.softmax_regression.softmax import SoftmaxRegression


def main():
    evaluate_for_train_months(4)
    #
    # train_data = get_data(datetime(2024, 1, 1), datetime(2024, 3, 1))
    # test_data = get_data(datetime(2024, 3, 1), datetime(2024, 3, 2))
    # model = SoftmaxRegression()
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


def evaluate_for_train_months(train_months):
    total = pd.DataFrame()
    train_start = datetime(2024, 1, 1)
    while train_start + relativedelta(months=train_months) <= datetime(2024, 12, 2):
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=1)

        train_data = get_data(train_start, train_end)
        test_data = get_data(test_start, test_end)

        model = SoftmaxRegression()
        model.train(train_data)
        predictions = model.predict(test_data)
        result = pd.DataFrame(predictions, columns=['DOWN', 'UP', 'NEUTRAL'])
        result[DataColumns.TARGET] = model.prepare_for_predict(test_data)[DataColumns.TARGET].values
        total = pd.concat([total, result], ignore_index=True)

        train_start = train_start + relativedelta(months=1)
    #
    __target_mapping = {'DOWN': 0, 'UP': 1, 'NEUTRAL': 2}
    test_y = [__target_mapping[target] for target in total[DataColumns.TARGET]]
    evaluate(total[['DOWN', 'UP', 'NEUTRAL']].values, np.array(test_y),
             {"train_size": train_months, "only2024": True},
             SoftmaxRegression().name())


if __name__ == "__main__":
    main()
