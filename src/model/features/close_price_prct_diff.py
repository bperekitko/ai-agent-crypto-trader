from pandas import DataFrame

from data.raw_data_columns import RawDataColumns
from model.features.feature import Feature


class CloseDiff(Feature):
    __NAME = 'close_percent_diff'

    def name(self):
        return self.__NAME

    def calculate(self, df: DataFrame):
        return df[RawDataColumns.CLOSE].pct_change() * 100
