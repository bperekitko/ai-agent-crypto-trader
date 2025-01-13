import json
import os
from typing import List

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Input
from keras.src.losses.losses import CategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate_for_highs_and_lows
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.hour_of_day import HourOfDaySine, HourOfDayCosine
from model.features.rsi import RSI
from model.features.target import Target, PercentileLabelingPolicy
from model.features.volume import Volume
from model.model import Model
from model.saved import SAVED_MODELS_PATH
from utils.log import get_logger


class Lstm(Model):
    __NAME = 'lstm'
    __LOG = get_logger(__NAME)

    def __init__(self):
        super().__init__()
        self.__version = 0.01
        self.__features: List[Feature] = [CloseDiff(), HighToClose(), CloseToLow(), Volume(), RSI(8), HourOfDaySine(), HourOfDayCosine()]
        neg_perc = 25
        pos_perc = 100 - neg_perc
        self.__target = Target(PercentileLabelingPolicy(neg_perc, pos_perc))
        self.params = {
            'loss_function_name': 'categorical_crossentropy',
            'sequence_window': 12,
            'layers': ['LSTM-100', 'LSTM-100', 'Dense-64', 'Dense-3'],
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 4,
            'early_stopping_metric': 'val_precision',
            'early_stopping_patience': 5,
            "target": f'Percentile_{neg_perc}_{pos_perc}',
            'features': [f.name() for f in self.__features]
        }

        self.__model = Sequential()

    def version(self) -> str:
        return f'{self.__version}'

    def name(self) -> str:
        return self.__NAME

    def describe(self) -> None:
        self.__LOG.info(f'{self.__NAME}')
        self.__LOG.info(json.dumps(self.params, indent=4))

    def predict(self, df: pd.DataFrame):
        train_data = self.prepare_data(df)
        x_train, _ = self.__to_sequences(train_data)
        return self.__model.predict(x_train)

    def train(self, df: pd.DataFrame):
        train_data = self.prepare_data(df)
        x_train, y_train = self.__to_sequences(train_data)

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
        self.__model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']), loss=self.params['loss_function_name'], metrics=['precision'])

        early_stopping = EarlyStopping(monitor=self.params['early_stopping_metric'], patience=self.params['early_stopping_patience'], restore_best_weights=True)
        self.__model.fit(x_train, y_train, epochs=self.params['epochs'], batch_size=self.params['batch_size'], validation_split=0.1, class_weight=class_weights_for_model,
                         callbacks=[early_stopping])

    def test(self, df: pd.DataFrame):
        self.__LOG.info(f'Testing model {self.__NAME}')
        probabilities = self.predict(df)
        train_data = self.prepare_data(df)
        evaluate_for_highs_and_lows(probabilities, self, train_data, self.__target.threshold_up, self.__target.threshold_down)

        # _, test_y = self.__to_sequences(train_data)
        # evaluate(probabilities, y_test, self.params, self.__NAME)

    def prepare_data(self, df):
        result_data = df[[DataColumns.DATE_CLOSE, DataColumns.HIGH, DataColumns.LOW, DataColumns.CLOSE]].copy()
        for feature in self.__features:
            result_data[feature.name()] = feature.calculate(df).values

        result_data[self.__target.name()] = self.__target.calculate(df)
        return result_data.dropna()

    def __to_sequences(self, train_data):
        x = train_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET, DataColumns.HIGH, DataColumns.LOW, DataColumns.CLOSE], axis=1).values
        y = train_data[DataColumns.TARGET]
        one_hot_encoded_y = to_categorical(y)
        return self.__create_sequences(x, one_hot_encoded_y)

    def __create_sequences(self, df, target):
        x, y = [], []
        for i in range(len(df) - self.params['sequence_window'] + 1):
            x.append(df[i:(i + self.params['sequence_window'])])
            y.append(target[i + self.params['sequence_window'] - 1])
        return np.array(x), np.array(y)

    def save(self):
        model_path = os.path.join(SAVED_MODELS_PATH, f'{self.name()}_{self.version()}.keras')
        self.__model.save(model_path)
        with open(os.path.join(SAVED_MODELS_PATH, f'{self.name()}_{self.version()}.json'), 'w') as file:
            json.dump(self.params, file, indent=4)

    def load(self, version: str):
        model_path = os.path.join(SAVED_MODELS_PATH, f'{self.name()}_{version}.h5')
        self.__model = load_model(model_path)
        self.__version = version
        return self
