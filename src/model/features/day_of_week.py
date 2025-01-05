import numpy as np
from pandas import DataFrame

from data.raw_data_columns import DataColumns
from model.features.feature import Feature


class DayOfWeekSine(Feature):
    def __init__(self):
        super().__init__('day_of_week_sine')

    def _calculate(self, df: DataFrame):
        copied = df.copy()
        copied['day'] = df[DataColumns.DATE_OPEN].dt.dayofweek
        return np.sin(2 * np.pi * copied['day'] / 7)


class DayOfWeekCosine(Feature):
    def __init__(self):
        super().__init__('day_of_week_cosine')

    def _calculate(self, df: DataFrame):
        copied = df.copy()
        copied['day'] = df[DataColumns.DATE_OPEN].dt.dayofweek
        return np.cos(2 * np.pi * copied['day'] / 7)
