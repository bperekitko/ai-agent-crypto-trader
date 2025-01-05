import json
from typing import List

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Input
from keras.src.losses.losses import CategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma import CloseToSma
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.rsi import RSI
from model.features.target import Target, PercentileLabelingPolicy
from model.model import Model
from utils.log import Logger


class Lstm(Model):
    __NAME = 'lstm'
    __LOG = Logger(__NAME)

    def __init__(self):
        self.__features: List[Feature] = [
            # AverageTrueRange(18),
            # Volume(),
            RSI(8),
            CloseDiff(),
            HighToClose(),
            CloseToLow(),
            CloseToSma(8),
        ]
        self.params = {
            'sequence_window': 12,
            'layers': ['LSTM-100', 'LSTM-100', 'Dense-64', 'Dense-3'],
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 4,
            'early_stopping_metric': 'val_precision',
            'early_stopping_patience': 5,
            'target_policy': 'percentile_50_50',
            'features': [f.name() for f in self.__features]
        }

        self.__model = Sequential()


        self.__target = Target(PercentileLabelingPolicy(50, 50))

    def name(self) -> str:
        return self.__NAME

    def describe(self) -> None:
        self.__LOG.info(f'{self.__NAME}')
        self.__LOG.info(json.dumps(self.params, indent=4))

    def predict(self, df: pd.DataFrame):
        x, y_true = self.prepare_data(df)
        return self.__model.predict(x)

    def train(self, df: pd.DataFrame):
        x_train, y_train = self.prepare_data(df)

        y_train_labels = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weights_for_model = dict(zip(np.unique(y_train_labels), class_weights))

        self.__model.add(Input(shape=(self.params['sequence_window'], len(self.__features))))
        self.__model.add(LSTM(units=100, return_sequences=True))
        self.__model.add(Dropout(0.2))
        self.__model.add(LSTM(units=100))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(units=64, activation='relu'))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(units=3, activation='softmax'))
        self.__model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']), loss=CategoricalCrossentropy, metrics=['precision'])


        early_stopping = EarlyStopping(monitor=self.params['early_stopping_metric'], patience=self.params['early_stopping_patience'], restore_best_weights=True)
        self.__model.fit(x_train, y_train, epochs=self.params['epochs'], batch_size=self.params['batch_size'], validation_split=0.1, class_weight=class_weights_for_model,
                         callbacks=[early_stopping])

    def test(self, df: pd.DataFrame):
        self.__LOG.info(f'Testing model {self.__NAME}')
        probabilities = self.predict(df)
        _, test_y = self.prepare_data(df)
        y_test = np.argmax(test_y, axis=1)

        evaluate(probabilities, y_test, self.params, self.__NAME)

    def prepare_data(self, df):
        train_data = df[[DataColumns.DATE_CLOSE]].copy()
        for feature in self.__features:
            train_data[feature.name()] = feature.calculate(df).values

        train_data[self.__target.name()] = self.__target.calculate(df)
        train_data.dropna(inplace=True)

        x = train_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET], axis=1).values
        y = train_data[DataColumns.TARGET]
        one_hot_encoded_y = to_categorical(y)
        return self.__create_sequences(x, one_hot_encoded_y)

    def __create_sequences(self, df, target):
        x, y = [], []
        for i in range(len(df) - self.params['sequence_window'] + 1):
            x.append(df[i:(i + self.params['sequence_window'])])
            y.append(target[i + self.params['sequence_window'] - 1])
        return np.array(x), np.array(y)

    def set_features(self, features):
        self.__features = features
        self.params['features'] = [f.name() for f in self.__features]
