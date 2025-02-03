import os
import pickle
from typing import List

import pandas as pd

from model.features.close_to_ema import CloseToEma
from model.features.ema_to_ema_ratio import EmaToEmaRatio

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)  # Displaying all columns when printing
pd.set_option('display.expand_frame_repr', False)  # Disable line wrap when printing

import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Input
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.src.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate
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

_LOG = get_logger('lstm')


class Lstm(Model):
    __NAME = 'lstm'

    def __init__(self):
        super().__init__()
        self.__version = 0.02
        self.__features: List[Feature] = [CloseDiff(), HighToClose(), CloseToLow(), Volume(), RSI(8), HourOfDaySine(), HourOfDayCosine(), CloseToEma(15)]
        neg_perc = 30
        pos_perc = 100 - neg_perc
        self.__target = Target(PercentileLabelingPolicy(neg_perc, pos_perc))
        self.params = {
            'loss_function_name': 'categorical_crossentropy',
            'sequence_window': 42,
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
        _LOG.info(f'{self.__NAME}')
        _LOG.info(self.params)
        for feature in self.__features:
            if hasattr(feature, 'is_fitted'):
                _LOG.debug(f'{feature.name()} fitted: {feature.is_fitted}')
        _LOG.debug(f'Target UP threshold: {self.__target.threshold_up(pd.DataFrame({"col": [1]})).values[0] * 100}%')
        _LOG.debug(f'DOWN: {self.__target.threshold_down(pd.DataFrame({"col": [1]})).values[0] * 100}%')

    def get_thresholds(self):
        target_up = self.__target.threshold_up(pd.DataFrame({'col': [1]})).values[0]
        target_down = self.__target.threshold_down(pd.DataFrame({'col': [1]})).values[0]
        return target_up, target_down

    def predict(self, df: pd.DataFrame):
        train_data = self.prepare_data(df)
        x_train, _ = self.to_sequences(train_data)

        target_up = self.__target.threshold_up(pd.DataFrame({'col': [1]})).values[0]
        target_down = self.__target.threshold_down(pd.DataFrame({'col': [1]})).values[0]

        return self.__model.predict(x_train), target_up, target_down

    def train(self, df: pd.DataFrame):
        _LOG.info(f'Training model, train data between: {df.head(1)[DataColumns.DATE_OPEN].values[0]} - {df.tail(1)[DataColumns.DATE_OPEN].values[0]}')
        self.params[
            'trained_on'] = f'{df.head(1)[DataColumns.DATE_OPEN].dt.strftime("%Y-%m-%d %H-%M").values[0]} - {df.tail(1)[DataColumns.DATE_OPEN].dt.strftime("%Y-%m-%d %H-%M").values[0]}'

        train_data = self.prepare_data(df)
        x_train, y_train = self.to_sequences(train_data)

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
        _LOG.info(f'Testing model {self.__NAME}')
        self.params[
            'tested_on'] = f'{df.head(1)[DataColumns.DATE_CLOSE].dt.strftime("%Y-%m-%d %H-%M").values[0]} - {df.tail(1)[DataColumns.DATE_CLOSE].dt.strftime("%Y-%m-%d %H-%M").values[0]}'
        probabilities, _, _ = self.predict(df)
        train_data = self.prepare_data(df)
        # evaluate_for_highs_and_lows(probabilities, self, train_data, self.__target.threshold_up, self.__target.threshold_down)

        _, test_y = self.to_sequences(train_data)
        self.params['adjusted_target'] = False
        y_true = np.argmax(test_y, axis=1)
        # evaluate(probabilities, y_true, self)
        return probabilities, y_true

    def prepare_data(self, df):
        result_data = df[[DataColumns.DATE_CLOSE, DataColumns.HIGH, DataColumns.LOW, DataColumns.CLOSE]].copy()
        for feature in self.__features:
            result_data[feature.name()] = feature.calculate(df).values

        result_data[self.__target.name()] = self.__target.calculate(df)
        return result_data.dropna()

    def to_sequences(self, train_data):
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
        data_to_serialize = {k: v for k, v in self.__dict__.items() if k != '__model'}
        with open(os.path.join(SAVED_MODELS_PATH, f'{self.name()}_{self.version()}_data.pkl'), 'wb') as file:
            pickle.dump(data_to_serialize, file)

    @classmethod
    def load(cls, version: str):
        model_path = os.path.join(SAVED_MODELS_PATH, f'{Lstm.__NAME}_{version}.keras')
        other_data_path = os.path.join(SAVED_MODELS_PATH, f'{Lstm.__NAME}_{version}_data.pkl')

        new_instance = cls()
        with open(other_data_path, 'rb') as file:
            other_data = pickle.load(file)
            new_instance.__dict__.update(other_data)
        new_instance.__model = load_model(model_path)

        return new_instance
