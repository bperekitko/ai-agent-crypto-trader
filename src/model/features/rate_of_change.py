import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model.features.feature import Feature


class RateOfChange(Feature):
    def __init__(self, period=12):
        super().__init__(f'rate_of_change_{period}')
        self.period = period
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        series = df['close'].pct_change(periods=self.period) * 100
        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        len_diff = len(df) - len(scaled)
        return pd.concat([pd.Series([np.nan] * len_diff), result]).reset_index(drop=True) if len_diff > 0 else result

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)
