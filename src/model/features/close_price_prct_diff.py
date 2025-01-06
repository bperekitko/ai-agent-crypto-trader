import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import winsorize_transform
from model.features.feature import Feature


class CloseDiff(Feature):

    def __init__(self):
        super().__init__('close_percent_diff')
        self.scaler = StandardScaler()
        self.winsor_min = None
        self.winsor_max = None
        self.is_fitted = False

    def _calculate(self, df: DataFrame) -> pd.Series:
        close_diff_series = df[DataColumns.CLOSE].pct_change() * 100
        close_diff_series.dropna(inplace=True)
        winsorized = self.__winsorize_values(close_diff_series)
        scaled_values = self.__scale_values(winsorized)
        self.is_fitted = True

        series = pd.Series(scaled_values.flatten())
        return pd.concat([pd.Series([np.nan] * (len(df) - len(scaled_values))), series]).reset_index(drop=True)

    def __winsorize_values(self, series: pd.Series):
        if self.is_fitted:
            return np.clip(series, a_min=self.winsor_min, a_max=self.winsor_max)
        else:
            result, self.winsor_min, self.winsor_max = winsorize_transform(series, 99, 1)
            return result

    def __scale_values(self, values: pd.Series):
        if self._bins > 0:
            values = self._binned_equal_size(values)
        values_reshaped_for_scaling = values.values.reshape(-1, 1)
        transformation_func = self.scaler.transform if self.is_fitted else self.scaler.fit_transform
        return transformation_func(values_reshaped_for_scaling)
