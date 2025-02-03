import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import numpy as np
from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class CloseToEma(Feature):
    def __init__(self, window=20):
        super().__init__(f'close_to_ema_{window}')
        self.scaler = StandardScaler()
        self.window = window
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False

    def _calculate(self, df: DataFrame):
        ema = df[DataColumns.CLOSE].ewm(span=self.window, adjust=False).mean()
        series = df[DataColumns.CLOSE] / ema
        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled_values = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        nans_to_add = self.window
        result = pd.Series(scaled_values.flatten())
        result.iloc[:nans_to_add] = np.nan
        return result

    def __scale_values(self, values):
        transformation_func = self.scaler.transform if self.is_fitted else self.scaler.fit_transform
        return transformation_func(values)
