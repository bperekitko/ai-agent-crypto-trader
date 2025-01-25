import os

import pandas as pd
import logging

from model.features.target import TargetLabel, Target
from model.softmax_regression.softmax import SoftmaxRegression

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)  # Displaying all columns when printing
pd.set_option('display.expand_frame_repr', False)  # Disable line wrap when printing

from model.lstm.lstm import Lstm

from datetime import datetime

from data.raw_data import get_data, get_last_x_intervals


def train():
    train_data = get_data(datetime(2024, 1, 1), datetime(2024, 11, 1))
    test_data = get_data(datetime(2024, 11, 1), datetime(2025, 1, 1))
    h = get_last_x_intervals(20)
    # model = Lstm()
    # model.train(train_data)
    # model.save()

    model = Lstm.load("0.01")
    # model.describe()
    # model.load("0.01")
    # h['percent_diff']=h['close'].pct_change() * 100
    # print(model.prepare_data(h))
    #
    predictions, up, down = model.predict(h.copy())
    print(f'{model.name()}: ')
    for i in range(len(predictions[0])):
        print(f'{TargetLabel(i).name}: {predictions[0][i] * 100}%')

    print(f'THRESHOLD UP: {up}')
    print(f'THRESHOLD DOWN: {down}')
    print(h)
    # model2 = SoftmaxRegression()
    # model2.load("0.01")
    # print(f'{model2.name()}: ')
    # print( model2.predict(h.copy()))


    # model.train(train_data)
    # model.test(test_data)
    # model.save()

if __name__ == "__main__":
    train()
