import json

from data.raw_data_columns import RawDataColumns
from model.features.analyze.analyze import winsorize_transform
from model.features.close_price_prct_diff import CloseDiff
from model.model import Model
from sklearn.preprocessing import StandardScaler

from model.softmax_regression import current_dir_file_path
from model.softmax_regression.scalers import save_scaler


class SoftmaxRegression(Model):
    __VERSION = 0.01
    __NAME = 'softmax_linear_regression'
    __FEATURES = [CloseDiff()]

    def __init__(self, df):
        self.df = df.copy()
        self.params = {
            "version": self.__VERSION
        }
        self.__prepare_features()
        self.__save_params()

    def name(self):
        return self.__NAME

    def describe(self):
        print(f'{self.__NAME}, v.{self.__VERSION}')
        print(self.params)
        columns = [RawDataColumns.DATE_OPEN, RawDataColumns.CLOSE, 'target', 'close_percent_diff']
        print(self.df[columns].describe())
        print(self.df[columns].tail(10))

    def predict(self):
        pass

    def train(self):
        pass

    def __prepare_features(self):
        self.__close_diff()

    def __close_diff(self):
        close_diff = CloseDiff()
        name = close_diff.name()
        self.df[name] = close_diff.calculate(self.df)
        self.df = self.df.dropna()
        self.df.loc[:, name], min_limit, max_limit = winsorize_transform(self.df[name])
        self.df = self.df.dropna()

        scaler = StandardScaler()
        self.df[name] = scaler.fit_transform(self.df[[name]])
        close_diff_scaler_name = save_scaler('close_diff_standard_scaler', scaler)
        self.params[f"{name}_winsorize_min"] = f'{min_limit}'
        self.params[f"{name}_winsorize_max"] = f'{max_limit}'
        self.params[f"{name}_standard_scaler_name"] = close_diff_scaler_name

    def __save_params(self):
        with open(current_dir_file_path(f'params-{self.__VERSION}.json'), 'w') as file:
            json.dump(self.params, file, indent=4)
