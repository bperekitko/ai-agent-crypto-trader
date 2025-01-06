import os

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Displaying all columns when printing
pd.set_option('display.max_columns', None)
# Disable line wrap when printing
pd.set_option('display.expand_frame_repr', False)

from model.features.target import TargetLabel, Target, PercentileLabelingPolicy
from model.lstm.lstm import Lstm

from model.softmax_regression.softmax import SoftmaxRegression
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from data.raw_data import get_data
from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate


def main():
    train_data = get_data(datetime(2024, 1, 1), datetime(2024, 11, 1))
    test_data = get_data(datetime(2024, 11, 1), datetime(2025, 1, 1))

    model = Lstm()
    model.train(train_data)
    model.test(test_data)


def evaluate_for_train_months(train_months, model_func):
    total = pd.DataFrame()
    train_start = datetime(2024, 1, 1)
    while train_start + relativedelta(months=train_months) <= datetime(2024, 12, 1):
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=1)

        train_data = get_data(train_start, train_end)
        test_data = get_data(test_start, test_end)

        model = model_func()
        model.train(train_data)
        predictions = model.predict(test_data)

        result = pd.DataFrame(predictions, columns=['DOWN', 'UP', 'NEUTRAL'])

        _, test_y = model.prepare_data(test_data)
        y_test = np.argmax(test_y, axis=1)

        result[DataColumns.TARGET] = y_test
        total = pd.concat([total, result], ignore_index=True)

        train_start = train_start + relativedelta(months=1)

    test_y = total[DataColumns.TARGET]
    evaluate(total[['DOWN', 'UP', 'NEUTRAL']].values, np.array(test_y), {"train_size": train_months}, model_func().name())


if __name__ == "__main__":
    main()
