import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.raw_data_columns import DataColumns
from model.features.analyze.analyze import box_cox_transform
from model.features.feature import Feature


class CumulatedVolume(Feature):
    def __init__(self, period=5):
        super().__init__(f'cumulated_volume_{period}')
        self.period= period
        self.scaler = StandardScaler()
        self.fitted_lambda = None
        self.box_cox_shift = None
        self.is_fitted = False

    def _calculate(self, df: pd.DataFrame):
        series = df[DataColumns.VOLUME].copy().rolling(window=self.period).sum()
        series = self.__transform(series.dropna())
        if self._bins > 0:
            series = self._binned_equal_size(series)

        scaled = self.__scale_values(series.reshape(-1, 1))
        self.is_fitted = True

        result = pd.Series(scaled.flatten())
        len_diff = len(df) - len(scaled)
        return pd.concat([pd.Series([np.nan] * len_diff), result]).reset_index(drop=True) if len_diff > 0 else result

    def __scale_values(self, values):
        return self.scaler.transform(values) if self.is_fitted else self.scaler.fit_transform(values)

    def __transform(self, series: pd.Series):
        if self.is_fitted:
            return box_cox_transform(series, self.fitted_lambda, self.box_cox_shift)
        else:
            result, self.fitted_lambda, self.box_cox_shift = box_cox_transform(series)
            return result