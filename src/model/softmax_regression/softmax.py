import json
from typing import List
from unittest.mock import inplace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import winsorize_transform, box_cox_transform
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma20 import CloseToSma20
from model.features.high_to_close import HighToClose
from model.model import Model
from model.softmax_regression import save_to_json, save_box_cox_lambda, save_scaler, load_scaler, load_box_cox_lambda
from model.softmax_regression.evaluation.evaluate import evaluate


class SoftmaxRegression(Model):
    __VERSION = 0.01
    __NAME = 'softmax_linear_regression'
    __features = []
    __target_mapping = {'DOWN': 0, 'UP': 1, 'NEUTRAL': 2}

    def __init__(self, df):
        self.params = {
            "version": self.__VERSION,
            "solver": "lbfgs",
            "max_iter": 200,
            "C": 1.0
        }
        self.model = LogisticRegression(solver=self.params['solver'], max_iter=self.params['max_iter'],
                                        C=self.params['C'])
        train_size = int(len(df) * 0.75)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        self.test_data = test_data[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()
        self.train_data = train_data[[DataColumns.DATE_CLOSE, DataColumns.TARGET]].copy()

        self.__prepare_train_data(train_data)
        self.__prepare_test_data(test_data)

        self.params['features_used'] = self.__features
        save_to_json(f'params_{self.__VERSION}', self.params)

    def name(self):
        return self.__NAME

    def describe(self):
        print(f'{self.__NAME}, v.{self.__VERSION}')
        print(json.dumps(self.params, indent=4))
        # for column in self.features:
        #     analyze(self.train_data[column], column)
        print(self.train_data.head(20))
        print(f'Test len: {len(self.train_data)}')

    def predict(self):
        pass

    def train(self):
        train_x = self.train_data[self.__features]
        train_y = [self.__target_mapping[target] for target in self.train_data[DataColumns.TARGET]]
        self.model.fit(train_x, train_y)

    def test(self):
        test_x = self.test_data[self.__features]
        test_y = [self.__target_mapping[target] for target in self.test_data[DataColumns.TARGET]]
        probabilities = self.model.predict_proba(test_x)
        evaluate(probabilities, test_y, self.params, self.__NAME, self.__VERSION)

    def __prepare_train_data(self, df):
        self.__close_diff(df)
        self.__high_to_close(df)
        self.__close_to_low(df)
        self.__close_to_sma20(df)
        for lag in range(1, 5):
            for feature in [CloseDiff().name(), HighToClose().name(), CloseToLow().name(), CloseToSma20().name()]:
                lagged_feature = f'{feature}_lag_{lag}'
                self.train_data[lagged_feature] = self.train_data[feature].shift(lag)
                self.__features.append(lagged_feature)
        self.train_data.dropna(inplace=True)

    def __prepare_test_data(self, test_df):
        self.__close_diff_from_test(test_df)
        self.__high_to_close_from_test(test_df)
        self.__close_to_low_from_test(test_df)
        self.__close_to_sma20_from_test(test_df)
        for lag in range(1, 5):
            for feature in [CloseDiff().name(), HighToClose().name(), CloseToLow().name(), CloseToSma20().name()]:
                lagged_feature = f'{feature}_lag_{lag}'
                self.test_data[lagged_feature] = self.test_data[feature].shift(lag)
                self.__features.append(lagged_feature)
        self.test_data.dropna(inplace=True)

    def __close_diff(self, df):
        feature = CloseDiff()
        name = feature.name()
        self.__features.append(name)
        self.train_data[name] = feature.calculate(df)
        self.train_data = self.train_data.dropna()
        self.train_data.loc[:, name], min_limit, max_limit = winsorize_transform(self.train_data[name])

        scaler = StandardScaler()
        self.train_data[name] = scaler.fit_transform(self.train_data[[name]])
        scaler_name = save_scaler(f'{name}_scaler_{self.__VERSION}', scaler)
        self.params[f"{name}_winsorize_min"] = min_limit
        self.params[f"{name}_winsorize_max"] = max_limit
        self.params[f"{name}_standard_scaler_name"] = scaler_name

    def __high_to_close(self, df):
        feature = HighToClose()
        name = feature.name()
        self.__features.append(name)
        self.train_data[name] = feature.calculate(df)
        self.train_data = self.train_data.dropna()
        self.train_data.loc[:, name], lambda_fitted, shift_value = box_cox_transform(self.train_data[name])

        scaler = StandardScaler()
        self.train_data[name] = scaler.fit_transform(self.train_data[[name]])
        scaler_name = save_scaler(f'{name}_scaler_{self.__VERSION}', scaler)
        lambda_name = save_box_cox_lambda(f'{name}_box_cox_lambda_{self.__VERSION}', lambda_fitted)
        self.params[f"{name}_standard_scaler_name"] = scaler_name
        self.params[f'{name}_box_cox_lambda_name'] = lambda_name
        self.params[f'{name}_box_cox_lambda_shift_value'] = shift_value

    def __close_to_low(self, df):
        feature = CloseToLow()
        name = feature.name()
        self.__features.append(name)
        self.train_data[name] = feature.calculate(df)
        self.train_data = self.train_data.dropna()
        self.train_data.loc[:, name], lambda_fitted, shift_value = box_cox_transform(self.train_data[name])

        scaler = StandardScaler()
        self.train_data[name] = scaler.fit_transform(self.train_data[[name]])
        scaler_name = save_scaler(f'{name}_scaler_{self.__VERSION}', scaler)
        lambda_name = save_box_cox_lambda(f'{name}_box_cox_lambda_{self.__VERSION}', lambda_fitted)
        self.params[f"{name}_standard_scaler_name"] = scaler_name
        self.params[f'{name}_box_cox_lambda_name'] = lambda_name
        self.params[f'{name}_box_cox_lambda_shift_value'] = shift_value

    def __close_to_sma20(self, df):
        feature = CloseToSma20()
        name = feature.name()
        self.__features.append(name)
        self.train_data[name] = feature.calculate(df)
        self.train_data = self.train_data.dropna()

        scaler = StandardScaler()
        self.train_data[name] = scaler.fit_transform(self.train_data[[name]])
        scaler_name = save_scaler(f'{name}_scaler_{self.__VERSION}', scaler)
        self.params[f"{name}_standard_scaler_name"] = scaler_name

    def __close_diff_from_test(self, df):
        feature = CloseDiff()
        name = feature.name()
        self.test_data[name] = feature.calculate(df)
        self.test_data = self.test_data.dropna()
        self.test_data.loc[:, name] = np.clip(self.test_data[name], a_min=self.params[f'{name}_winsorize_min'],
                                              a_max=self.params[f'{name}_winsorize_max'])
        scaler = load_scaler(self.params[f'{name}_standard_scaler_name'])
        self.test_data[name] = scaler.transform(self.test_data[[name]])

    def __high_to_close_from_test(self, test_df):
        feature = HighToClose()
        name = feature.name()
        self.test_data[name] = feature.calculate(test_df)
        self.test_data = self.test_data.dropna()
        box_cox_lambda = load_box_cox_lambda(self.params[f'{name}_box_cox_lambda_name'])
        shift_value = self.params[f'{name}_box_cox_lambda_shift_value']
        self.test_data.loc[:, name] = box_cox_transform(self.test_data[name], box_cox_lambda, shift_value)

        scaler = load_scaler(self.params[f'{name}_standard_scaler_name'])
        self.test_data[name] = scaler.transform(self.test_data[[name]])

    def __close_to_low_from_test(self, test_df):
        feature = CloseToLow()
        name = feature.name()
        self.test_data[name] = feature.calculate(test_df)
        self.test_data = self.test_data.dropna()
        box_cox_lambda = load_box_cox_lambda(self.params[f'{name}_box_cox_lambda_name'])
        shift_value = self.params[f'{name}_box_cox_lambda_shift_value']
        self.test_data.loc[:, name] = box_cox_transform(self.test_data[name], box_cox_lambda, shift_value)

    def __close_to_sma20_from_test(self, test_df):
        feature = CloseToSma20()
        name = feature.name()
        self.test_data[name] = feature.calculate(test_df)
        self.test_data = self.test_data.dropna()
        scaler = load_scaler(self.params[f'{name}_standard_scaler_name'])

        self.test_data[name] = scaler.transform(self.test_data[[name]])
