import os
import pickle
from typing import List

import pandas as pd

from model.evaluation.evaluate_binary_model import evaluate_binary_model, evaluate_simple_stats
from model.features.atr import AverageTrueRange
from model.features.bollinger_bands import BollingerBandsWidth
from model.features.close_to_ema import CloseToEma
from model.features.ema_to_ema_ratio import EmaToEmaRatio
from model.features.macd import MacdSignal, MacdLine, MacdHistogram
from model.features.stochastic_oscillator import StochasticOscillator

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pd.set_option('display.max_columns', None)  # Displaying all columns when printing
pd.set_option('display.expand_frame_repr', False)  # Disable line wrap when printing

import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from sklearn.utils.class_weight import compute_class_weight

from data.raw_data_columns import DataColumns
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.hour_of_day import HourOfDaySine, HourOfDayCosine
from model.features.rsi import RSI
from model.features.target import HighAboveThreshold, LowAboveThreshold
from model.features.volume import Volume
from model.model import Model
from model.saved import SAVED_MODELS_PATH
from utils.log import get_logger
from abc import ABC

_LOG = get_logger('binary_lstm')


class BinaryLstm(Model, ABC):
    def __init__(self, target: Feature, name):
        super().__init__()
        self.__version = 0.01
        self.__name = name
        self.features: List[Feature] = [CloseDiff(), HighToClose(), CloseToLow(), Volume(), RSI(8), HourOfDaySine(), HourOfDayCosine(), AverageTrueRange(14),
                                        BollingerBandsWidth(20, 2), MacdSignal(), MacdLine(), MacdHistogram(), StochasticOscillator()]
        self.params = {
            'sequence_window': 24,
            'learning_rate': 0.001,
            'batch_size': 32,
            'features': [f.name() for f in self.features]
        }

        self.__target: Feature = target
        self.__model = Sequential()

    def version(self) -> str:
        return f'{self.__version}'

    def name(self) -> str:
        return self.__name

    def predict(self, df: pd.DataFrame):
        train_data = self.prepare_data(df)
        x_train, y_true = self.to_sequences(train_data)
        return self.__model.predict(x_train), y_true

    def test(self, df: pd.DataFrame):
        probabilities, _ = self.predict(df)
        train_data = self.prepare_data(df)
        _, true_y = self.to_sequences(train_data)

        evaluate_binary_model(probabilities, true_y, self)
        return probabilities, true_y

    def train(self, df: pd.DataFrame):
        _LOG.info(f'Starting to train: {self.name()}')
        split_index = int(len(df) * 0.85)
        train_df = df.iloc[:split_index].copy()
        val_df = df.iloc[split_index:].copy()

        train_data = self.prepare_data(train_df)
        val_data = self.prepare_data(val_df)

        x_train, y_train = self.to_sequences(train_data)
        x_val, y_val = self.to_sequences(val_data)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_for_model = dict(zip(np.unique(y_train), class_weights))

        self.init_model()
        early_stopping = EarlyStopping(monitor='val_precision', patience=6, restore_best_weights=True, mode='max')
        self.__model.fit(x_train, y_train, epochs=100, batch_size=self.params['batch_size'], validation_data=(x_val, y_val), class_weight=class_weights_for_model,
                         callbacks=[early_stopping])
        _LOG.info(f'{self.name()} trained')

    def prepare_data(self, df):
        result_data = df[[DataColumns.DATE_CLOSE]].copy()

        for feature in self.features:
            result_data[feature.name()] = feature.calculate(df).values

        result_data[self.__target.name()] = self.__target.calculate(df)
        return result_data.dropna()

    def to_sequences(self, train_data):
        x = train_data.drop([DataColumns.DATE_CLOSE, self.__target.name()], axis=1).values
        y = train_data[self.__target.name()].values
        return self.__create_sequences(x, y)

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

    def init_model(self):
        self.__model.add(Input(shape=(self.params['sequence_window'], len(self.features))))
        self.__model.add(LSTM(units=100))
        self.__model.add(Dropout(0.2))
        self.__model.add(BatchNormalization())
        self.__model.add(Dense(units=32, activation='relu'))
        self.__model.add(Dropout(0.2))
        self.__model.add(Dense(1, activation='sigmoid'))
        self.__model.compile(optimizer=Adam(learning_rate=self.params['learning_rate']), loss='binary_crossentropy', metrics=['precision'])


# Tries to predict whether price of next candle will raise above certain threshold (in percent)
class LongHighPriceLstm(BinaryLstm):
    NAME = 'long_high_price_lstm'

    def __init__(self):
        target = HighAboveThreshold(0.5)
        super().__init__(target, name=LongHighPriceLstm.NAME)

    @classmethod
    def load(cls, version: str):
        model_path = os.path.join(SAVED_MODELS_PATH, f'{LongHighPriceLstm.NAME}_{version}.keras')
        other_data_path = os.path.join(SAVED_MODELS_PATH, f'{LongHighPriceLstm.NAME}_{version}_data.pkl')

        new_instance = cls()
        with open(other_data_path, 'rb') as file:
            other_data = pickle.load(file)
            new_instance.__dict__.update(other_data)
        new_instance.__model = load_model(model_path)

        return new_instance


# Tries to predict whether price will not decrease (reaches low) of certain threshold (in percent)
class LongLowPriceLstm(BinaryLstm):
    NAME = 'long_low_price_lstm'

    def __init__(self):
        target = LowAboveThreshold(0.5)
        super().__init__(target, name=LongLowPriceLstm.NAME)

    @classmethod
    def load(cls, version: str):
        model_path = os.path.join(SAVED_MODELS_PATH, f'{LongLowPriceLstm.NAME}_{version}.keras')
        other_data_path = os.path.join(SAVED_MODELS_PATH, f'{LongLowPriceLstm.NAME}_{version}_data.pkl')

        new_instance = cls()
        with open(other_data_path, 'rb') as file:
            other_data = pickle.load(file)
            new_instance.__dict__.update(other_data)
        new_instance.__model = load_model(model_path)

        return new_instance


class LongTradeLstm(Model):
    def __init__(self, high_model: LongHighPriceLstm, low_model: LongLowPriceLstm):
        super().__init__()
        self.high_model = high_model
        self.low_model = low_model

    def name(self) -> str:
        return 'long_trade_lstm'

    def version(self) -> str:
        return '0.01'

    def predict(self, df: pd.DataFrame):
        high_predictions, high_true = self.high_model.predict(df)
        low_predictions, low_true = self.low_model.predict(df)
        y_pred_high = (high_predictions > 0.5).astype(int).astype(float).flatten()
        y_pred_low = (low_predictions > 0.5).astype(int).astype(float).flatten()

        df = pd.DataFrame({
            'high_pred': y_pred_high,
            'low_pred': y_pred_low,
        })

        df['long_pred'] = df['high_pred'] * df['low_pred']
        return df['long_pred']

    def train(self, df: pd.DataFrame):
        self.high_model.train(df)
        self.low_model.train(df)

    def test(self, df: pd.DataFrame):
        high_predictions, high_true = self.high_model.predict(df)
        low_predictions, low_true = self.low_model.predict(df)
        y_pred_high = (high_predictions > 0.5).astype(int).astype(float).flatten()
        y_pred_low = (low_predictions > 0.5).astype(int).astype(float).flatten()

        df = pd.DataFrame({
            'high_pred': y_pred_high,
            'low_pred': y_pred_low,
            'high_true': high_true,
            'low_true': low_true
        })

        df['long_pred'] = df['high_pred'] * df['low_pred']
        df['y_true'] = df['high_true'] * df['low_true']

        evaluate_simple_stats(df['long_pred'], df['y_true'], f'{self.name()}_{self.version()}')

    def save(self):
        self.high_model.save()
        self.low_model.save()

    def load(self, version: str):
        pass
