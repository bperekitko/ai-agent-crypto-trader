import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model.features.feature import Feature


class CommodityChannelIndex(Feature):
    def __init__(self, period=20):
        super().__init__(f'CCI_{period}')
        self.period = period
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['sma_tp'] = df['typical_price'].rolling(window=self.period).mean()

        df['mad'] = df['typical_price'].rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        series = (df['typical_price'] - df['sma_tp']) / (0.015 * df['mad'])

        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        return pd.Series(scaled.flatten())

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)
