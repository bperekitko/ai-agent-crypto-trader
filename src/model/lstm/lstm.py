from typing import List

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.utils import to_categorical

from data.raw_data_columns import DataColumns
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma import CloseToSma
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.rsi import RSI
from model.features.volume import Volume
from model.model import Model
from utils.log import Logger


class Lstm(Model):
    __NAME = 'lstm'
    __target_mapping = {'DOWN': 0, 'UP': 1, 'NEUTRAL': 2}
    __LOG = Logger(__NAME)

    def __init__(self):
        self.__features: List[Feature] = [
            Volume(),
            RSI(8),
            CloseDiff(),
            HighToClose(),
            CloseToLow(),
            CloseToSma(8),
        ]
        self.params = {

        }
        self.model = Sequential()

    def name(self) -> str:
        return self.__NAME

    def describe(self) -> None:
        pass

    def predict(self, df: pd.DataFrame):
        pass

    def train(self, df: pd.DataFrame):
        train_data = self.__prepare_train_data(df)
        x = train_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET], axis=1).values
        y = [self.__target_mapping[target] for target in train_data[DataColumns.TARGET]]
        one_hot_encoded_y = to_categorical(y)
        print(one_hot_encoded_y)
        x_train, y_train = self.__create_sequences(x, one_hot_encoded_y, 18)
        # print(x_train)
        for seq in x_train:
            print(f'A SINGLE SEQ', seq)
        print(y_train == one_hot_encoded_y)

    def __prepare_train_data(self, df) -> pd.DataFrame:
        train_data = df[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()
        for feature in self.__features:
            train_data[feature.name()] = feature.calculate(df).values

        train_data.dropna(inplace=True)
        return train_data

    def __create_sequences(self, df, target, seq_length):
        x, y = [], []
        print(f'DF LEN: {len(df)}')
        for i in range(len(df) - seq_length + 1):
            x.append(df[i:(i + seq_length)])
            y.append(target[i + seq_length - 1])
        return np.array(x), np.array(y)

    def prepare_for_predict(self, df):
        data = df[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()
        for feature in self.__features:
            data[feature.name()] = feature.calculate(df).values
        data.dropna(inplace=True)
        return data
