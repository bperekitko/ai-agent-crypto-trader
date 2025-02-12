import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class StochasticOscillator(Feature):
    def __init__(self, period=14, smoothing_avg=3):
        super().__init__(f'stochastic_oscillator_{period}_{smoothing_avg}')
        self.period = period
        self.smoothing_avg = smoothing_avg
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        df['lowest_low'] = df[DataColumns.LOW].rolling(window=self.period).min()
        df['highest_high'] = df[DataColumns.HIGH].rolling(window=self.period).max()
        df['stoch_K'] = 100 * (df[DataColumns.CLOSE] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
        series = df['stoch_K'].rolling(window=self.smoothing_avg).mean()

        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        return pd.Series(scaled.flatten())

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)
