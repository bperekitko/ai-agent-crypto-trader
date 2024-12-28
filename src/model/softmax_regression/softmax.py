import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import RawDataColumns
from model.features.analyze.analyze import winsorize_transform, box_cox_transform
from model.features.close_price_prct_diff import CloseDiff
from model.features.close_to_low import CloseToLow
from model.features.close_to_sma20 import CloseToSma20
from model.features.high_to_close import HighToClose
from model.model import Model
from model.softmax_regression import save_to_json, save_box_cox_lambda, save_scaler, load_scaler, load_box_cox_lambda


class SoftmaxRegression(Model):
    __VERSION = 0.01
    __NAME = 'softmax_linear_regression'

    def __init__(self, df):
        train_size = int(len(df) * 0.8)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]

        self.test_data = test_data[[RawDataColumns.DATE_CLOSE]].copy()
        self.train_data = train_data[[RawDataColumns.DATE_CLOSE]].copy()

        self.params = {
            "version": self.__VERSION
        }

        self.__prepare_train_data(train_data)
        self.__prepare_test_data(test_data)

        save_to_json(f'params_{self.__VERSION}', self.params)

    def name(self):
        return self.__NAME

    def describe(self):
        print(f'{self.__NAME}, v.{self.__VERSION}')
        print(json.dumps(self.params, indent=4))
        # columns = ['close_to_sma_20', 'close_percent_diff', 'high_to_close_diff', 'close_to_low_percent']
        # for column in columns:
        #     analyze(self.train_data[column], column)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)  # Wyłączanie łamania linii
        print(self.test_data.tail(20))
        print(f'Test len: {len(self.test_data)}')

    def predict(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def __prepare_train_data(self, df):
        self.__close_diff(df)
        self.__high_to_close(df)
        self.__close_to_low(df)
        self.__close_to_sma20(df)

    def __prepare_test_data(self, test_df):
        self.__close_diff_from_test(test_df)
        self.__high_to_close_from_test(test_df)
        self.__close_to_low_from_test(test_df)
        self.__close_to_sma20_from_test(test_df)

    def __close_diff(self, df):
        feature = CloseDiff()
        name = feature.name()
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
