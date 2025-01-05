import numpy as np
from pandas import DataFrame

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class HourOfDayCosine(Feature):
    def __init__(self):
        super().__init__('hour_cosine')

    def _calculate(self, df: DataFrame):
        copied = df.copy()
        copied['hour'] = df[DataColumns.DATE_OPEN].dt.hour
        return np.cos(2 * np.pi * copied['hour'] / 24)


class HourOfDaySine(Feature):
    def __init__(self):
        super().__init__('hour_sine')

    def _calculate(self, df: DataFrame):
        copied = df.copy()
        copied['hour'] = df[DataColumns.DATE_OPEN].dt.hour
        return np.sin(2 * np.pi * copied['hour'] / 24)
