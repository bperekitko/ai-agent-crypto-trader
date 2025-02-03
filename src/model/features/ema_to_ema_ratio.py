import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class EmaToEmaRatio(Feature):
    def __init__(self, long_ema_window: int, short_ema_window: int):
        super().__init__(f'ema_{long_ema_window}_to_ema_{short_ema_window}')
        self.long_ema_window = long_ema_window
        self.short_ema_window = short_ema_window
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        short_ema = df[DataColumns.CLOSE].ewm(span=self.short_ema_window, adjust=False).mean()
        long_ema = df[DataColumns.CLOSE].ewm(span=self.long_ema_window, adjust=False).mean()

        ratio = long_ema / short_ema
        scaled_values = self.__scale_values(ratio.values.reshape(-1, 1))
        self.is_fitted = True

        nans_to_add = self.long_ema_window
        result = pd.Series(scaled_values.flatten())
        result.iloc[:nans_to_add] = np.nan
        return result

    def __scale_values(self, values):
        transformation_func = self.scaler.transform if self.is_fitted else self.scaler.fit_transform
        return transformation_func(values)
