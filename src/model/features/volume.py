import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature


class Volume(Feature):
    def __init__(self):
        super().__init__('volume')
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: DataFrame):
        values = df[DataColumns.VOLUME]
        transformed = self.__transform(values)
        scaled = self.__scale(transformed)
        self.is_fitted = True
        return pd.Series(scaled.flatten())

    def __scale(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return box_cox_transform(series, self.fitted_lambda, self.box_cox_shift).reshape(-1, 1)
        else:
            result, self.fitted_lambda, self.box_cox_shift = box_cox_transform(series)
            return result.reshape(-1, 1)
