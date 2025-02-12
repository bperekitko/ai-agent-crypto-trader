import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class BollingerBandsWidth(Feature):
    def __init__(self, window=20, multiplier=2):
        super().__init__('bollinger_bands_width')
        self.window = window
        self.multiplier = multiplier
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, input_df: pd.DataFrame):
        df = input_df.copy()
        df['ma_bb'] = df[DataColumns.CLOSE].rolling(window=self.window).mean()
        df['std_bb'] = df[DataColumns.CLOSE].rolling(window=self.window).std()
        df['upper_bb'] = df['ma_bb'] + self.multiplier * df['std_bb']
        df['lower_bb'] = df['ma_bb'] - self.multiplier * df['std_bb']
        series = (df['upper_bb'] - df['lower_bb']) / df['ma_bb']

        if self._bins > 0:
            series = self._binned_equal_size(series)
        scaled = self.__scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        nans_to_add = self.window
        result.iloc[:nans_to_add] = np.nan
        return result

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)
