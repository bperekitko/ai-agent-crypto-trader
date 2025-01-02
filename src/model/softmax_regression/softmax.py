import json
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data.raw_data_columns import DataColumns
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma import CloseToSma
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.rsi import RSI
from model.features.volume import Volume
from model.model import Model
from model.softmax_regression.evaluation.evaluate import evaluate
from utils.log import Logger


class SoftmaxRegression(Model):
    __NAME = 'softmax_linear_regression'
    __target_mapping = {'DOWN': 0, 'UP': 1, 'NEUTRAL': 2}
    __LOG = Logger(__NAME)

    def __init__(self):
        self.__features: List[Feature] = [
            Volume(),
            RSI(8),
            # RSI(30),
            CloseDiff(),
            # CloseDiff().lagged(1),
            # CloseDiff().lagged(2),
            # CloseDiff().lagged(3),
            # CloseDiff().lagged(4),
            HighToClose(),
            # HighToClose().lagged(1),
            # HighToClose().lagged(2),
            # HighToClose().lagged(3),
            # HighToClose().lagged(4),
            # HighToClose().lagged(5),
            CloseToLow(),
            # CloseToLow().lagged(1),
            # CloseToLow().lagged(2),
            # CloseToLow().lagged(3),
            # CloseToLow().lagged(4),
            CloseToSma(8),
            # CloseToSma(8).lagged(1)
        ]
        self.params = {
            "solver": "lbfgs",
            "max_iter": 200,
            "C": 1.0,
            "features": [feature.name() for feature in self.__features]
        }
        self.model = LogisticRegression(solver=self.params['solver'], max_iter=self.params['max_iter'],
                                        C=self.params['C'])

    def name(self):
        return self.__NAME

    def describe(self):
        self.__LOG.info(f'{self.__NAME}')
        self.__LOG.info(json.dumps(self.params, indent=4))

    def predict(self, df: pd.DataFrame):
        input_data = self.prepare_for_predict(df)
        self.__LOG.info(f'Creating predictions for input of length: {len(df)}')
        test_x = input_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET], axis=1)
        result = self.model.predict_proba(test_x)
        self.__LOG.info(f'Predictions created. Size: {len(result)}')
        return result

    def train(self, df: pd.DataFrame):
        train_data = self.__prepare_train_data(df)
        train_x = train_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET], axis=1)
        train_y = [self.__target_mapping[target] for target in train_data[DataColumns.TARGET]]

        self.__LOG.info(f'Starting to train model {self.__NAME}')
        self.model.fit(train_x, train_y)
        self.__LOG.info(f'Model {self.__NAME} successfully trained')

    def test(self, df: pd.DataFrame):
        self.__LOG.info(f'Testing model {self.__NAME}')
        probabilities = self.predict(df)
        test_y = [self.__target_mapping[target] for target in self.prepare_for_predict(df)[DataColumns.TARGET]]
        evaluate(probabilities, np.array(test_y), self.params, self.__NAME)

    def __prepare_train_data(self, df) -> pd.DataFrame:
        train_data = df[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()
        for feature in self.__features:
            train_data[feature.name()] = feature.calculate(df).values

        train_data.dropna(inplace=True)
        return train_data

    def prepare_for_predict(self, df):
        data = df[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()
        for feature in self.__features:
            data[feature.name()] = feature.calculate(df).values
        data.dropna(inplace=True)
        return data

    def test_different_sma(self, window, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.__features = [CloseDiff(), HighToClose(), CloseToLow(), CloseToSma(window)]
        self.params['sma_window'] = window
        self.train(train_df)
        self.test(test_df)

    def test_different_lags(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        lagging_close_diff = [
            CloseDiff().lagged(1),
            CloseDiff().lagged(2),
            CloseDiff().lagged(3),
            CloseDiff().lagged(4),
            CloseDiff().lagged(5),
        ]

        lagging_high = [
            HighToClose().lagged(1),
            HighToClose().lagged(2),
            HighToClose().lagged(3),
            HighToClose().lagged(4),
            HighToClose().lagged(5),
        ]

        lagging_close_to_low = [
            CloseToLow().lagged(1),
            CloseToLow().lagged(2),
            CloseToLow().lagged(3),
            CloseToLow().lagged(4),
            CloseToLow().lagged(5),
        ]

        lagging_close_to_smaf = [
            CloseToSma(8).lagged(1),
            CloseToSma(8).lagged(2),
            CloseToSma(8).lagged(3),
            CloseToSma(8).lagged(4),
            CloseToSma(8).lagged(5),
        ]

        prefixy1 = [lagging_close_diff[:i] for i in range(1, 6)]
        prefixy2 = [lagging_close_to_low[:i] for i in range(1, 6)]
        prefixy3 = [lagging_high[:i] for i in range(1, 6)]
        prefixy4 = [lagging_close_to_smaf[:i] for i in range(1, 6)]

        # Użycie product, aby utworzyć kombinacje z prefixów każdej tablicy
        combinations = list(product(prefixy1, prefixy2, prefixy3, prefixy4))

        # Wyświetlenie wszystkich kombinacji
        for kombinacja in combinations:
            features = [element for sublist in kombinacja for element in sublist]
            self.__features = [CloseDiff(),
                               HighToClose(),
                               CloseToLow(),
                               CloseToSma(8)] + features

            self.params['lagged_features'] = [feat.name() for feat in features]
            self.train(train_df)
            self.test(test_df)

    def test_different_rsi(self, window, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.__features += [RSI(window)]
        self.params['RSI WINDOW'] = window
        self.train(train_df)
        self.test(test_df)

    def test_volume_lagged(self, lags, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.__features = [Volume().lagged(i) for i in range(1, lags + 1)] + self.__features
        self.params["features"] = [feature.name() for feature in self.__features]
        self.train(train_df)
        self.test(test_df)
