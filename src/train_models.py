import os

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)  # Displaying all columns when printing
pd.set_option('display.expand_frame_repr', False)  # Disable line wrap when printing

from model.lstm.lstm import Lstm

from datetime import datetime

from data.raw_data import get_data

def train():
    train_data = get_data(datetime(2024, 1, 1), datetime(2024, 11, 1))
    test_data = get_data(datetime(2024, 11, 1), datetime(2025, 1, 1))

    # model = Lstm()
    # model.train(train_data)
    # model.test(test_data)
    # model.save()

if __name__ == "__main__":
    train()
