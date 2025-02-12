from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class Macd(Feature, ABC):
    HISTOGRAM_SERIES = 'macd_hist'
    MACD_SERIES = 'macd'
    SIGNAL_SERIES = 'macd_signal'

    def __init__(self, name, first_ema_window=12, second_ema_window=26, signal_window=9):
        super().__init__(name)
        self.first_ema_window = first_ema_window
        self.second_ema_window = second_ema_window
        self.signal_window = signal_window
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, input_df: pd.DataFrame):
        df = input_df.copy()

        ema12 = df[DataColumns.CLOSE].ewm(span=self.first_ema_window, adjust=False).mean()
        ema26 = df[DataColumns.CLOSE].ewm(span=self.second_ema_window, adjust=False).mean()
        df['ema12'] = ema12
        df['ema26'] = ema26
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=self.signal_window, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        series = df[self.series_name()]

        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.scale_values(series.values.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        nans_to_add = self.second_ema_window + self.signal_window
        result.iloc[:nans_to_add] = np.nan
        return result

    def scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    @abstractmethod
    def series_name(self) -> str:
        pass


class MacdHistogram(Macd):
    def __init__(self, first_ema_window=12, second_ema_window=26, signal_window=9):
        super().__init__(f'{Macd.HISTOGRAM_SERIES}', first_ema_window, second_ema_window, signal_window)

    def series_name(self) -> str:
        return Macd.HISTOGRAM_SERIES


class MacdSignal(Macd):
    def __init__(self, first_ema_window=12, second_ema_window=26, signal_window=9):
        super().__init__(f'{Macd.SIGNAL_SERIES}_{signal_window}', first_ema_window, second_ema_window, signal_window)

    def series_name(self) -> str:
        return Macd.SIGNAL_SERIES


class MacdLine(Macd):
    def __init__(self, first_ema_window=12, second_ema_window=26, signal_window=9):
        super().__init__(f'{Macd.MACD_SERIES}_ema{first_ema_window}_ema{second_ema_window}', first_ema_window, second_ema_window, signal_window)

    def series_name(self) -> str:
        return Macd.MACD_SERIES
