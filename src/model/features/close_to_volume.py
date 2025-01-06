import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature


class CloseToVolume(Feature):
    def __init__(self):
        super().__init__('close_to_volume')
        self.__scaler = StandardScaler()
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False

    def _calculate(self, df: DataFrame):
        values = df[DataColumns.CLOSE] / df[DataColumns.VOLUME]
        transformed = self.__transform(values)
        scaled = self.__scale(transformed)
        self.is_fitted = True
        return pd.Series(scaled.flatten())

    def __scale(self, values):
        if self._bins > 0:
            values = self._binned_equal_size(values)
        return self.__scaler.transform(values.reshape(-1, 1)) if self.is_fitted else self.__scaler.fit_transform(values.reshape(-1, 1))

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return box_cox_transform(series, self.fitted_lambda, self.box_cox_shift)
        else:
            result, self.fitted_lambda, self.box_cox_shift = box_cox_transform(series)
            return result
