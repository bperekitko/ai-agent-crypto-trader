import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import winsorize_transform
from model.features.feature import Feature


class HighPercentageChange(Feature):
    def __init__(self):
        super().__init__('high_pct_change')
        self.scaler = StandardScaler()
        self.winsor_min = None
        self.winsor_max = None
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        series = df[DataColumns.HIGH].pct_change()
        series = self.__transform(series.dropna())
        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        len_diff = len(df) - len(scaled)
        return pd.concat([pd.Series([np.nan] * len_diff), result]).reset_index(drop=True) if len_diff > 0 else result

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return np.clip(series, a_min=self.winsor_min, a_max=self.winsor_max)
        else:
            result, self.winsor_min, self.winsor_max = winsorize_transform(series, 99, 1)
            return result
