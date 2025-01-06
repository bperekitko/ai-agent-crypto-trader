import numpy as np
import pandas as pd
import ta.momentum
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class RSI(Feature):
    def __init__(self, window):
        super().__init__(f'rsi_{window}')
        self.window = window
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: DataFrame):
        series = ta.momentum.RSIIndicator(df[DataColumns.CLOSE], window=self.window).rsi()
        series.dropna(inplace=True)
        if self._bins > 0:
            series = self._binned_equal_size(series)
        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        return pd.concat([pd.Series([np.nan] * (len(df) - len(scaled))), result]).reset_index(drop=True)

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)
