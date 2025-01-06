import json
from typing import List
from unittest.mock import inplace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from data.raw_data_columns import DataColumns
from model.evaluation.evaluate import evaluate, evaluate_for_highs_and_lows
from model.features.atr import AverageTrueRange
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma import CloseToSma
from model.features.feature import Feature
from model.features.high_to_close import HighToClose
from model.features.hour_of_day import HourOfDaySine, HourOfDayCosine
from model.features.rsi import RSI
from model.features.target import Target, PercentileLabelingPolicy
from model.features.volume import Volume
from model.model import Model
from utils.log import Logger


class SoftmaxRegression(Model):
    __NAME = 'softmax_linear_regression'
    __LOG = Logger(__NAME)

    def __init__(self):
        self.__features: List[Feature] = [AverageTrueRange(18), Volume(), RSI(8), CloseDiff(), HighToClose(), CloseToLow(), CloseToSma(8), HourOfDaySine(), HourOfDayCosine()]
        neg_perc = 10
        pos_perc = 100 - neg_perc
        self.__target = Target(PercentileLabelingPolicy(neg_perc, pos_perc))
        self.params = {
            "solver": "lbfgs",
            "max_iter": 200,
            "C": 1.0,
            "features": [feature.name() for feature in self.__features],
            "target": f'Percentile_{neg_perc}_{pos_perc}'
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
        test_x = input_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET, DataColumns.HIGH, DataColumns.LOW, DataColumns.CLOSE], axis=1)
        result = self.model.predict_proba(test_x)
        self.__LOG.info(f'Predictions created. Size: {len(result)}')
        return result

    def train(self, df: pd.DataFrame):
        train_data = self.__prepare_train_data(df)
        train_x = train_data.drop([DataColumns.DATE_CLOSE, DataColumns.TARGET], axis=1)
        train_y = train_data[DataColumns.TARGET]

        self.__LOG.info(f'Starting to train model {self.__NAME}')

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
        class_weights_for_model = dict(zip(np.unique(train_y), class_weights))
        self.model = LogisticRegression(class_weight=class_weights_for_model, solver=self.params['solver'],
                                        max_iter=self.params['max_iter'],
                                        C=self.params['C'])

        self.model.fit(train_x, train_y)
        self.__LOG.info(f'Model {self.__NAME} successfully trained')

    def test(self, df: pd.DataFrame):
        self.__LOG.info(f'Testing model {self.__NAME}')
        probabilities = self.predict(df)
        input_data = self.prepare_for_predict(df)

        evaluate_for_highs_and_lows(probabilities, self.params, self.__NAME, input_data, self.__target.threshold_up, self.__target.threshold_down)
        # evaluate(probabilities, input_data[DataColumns.TARGET], self.params, self.__NAME)

    def __prepare_train_data(self, df) -> pd.DataFrame:
        train_data = df[[DataColumns.DATE_CLOSE]].copy()
        for feature in self.__features:
            train_data[feature.name()] = feature.calculate(df).values
        train_data[self.__target.name()] = self.__target.calculate(df)
        train_data.dropna(inplace=True)
        return train_data

    def prepare_for_predict(self, df):
        data = df[[DataColumns.DATE_CLOSE, DataColumns.HIGH, DataColumns.LOW, DataColumns.CLOSE]].copy()
        for feature in self.__features:
            data[feature.name()] = feature.calculate(df).values
        data[self.__target.name()] = self.__target.calculate(df)
        return data.dropna()

    def set_features(self, features):
        self.__features = features
        self.params['features'] = [f.name() for f in self.__features]
