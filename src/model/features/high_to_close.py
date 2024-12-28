from pandas import DataFrame

from data.raw_data_columns import RawDataColumns
from model.features.feature import Feature


class HighToClose(Feature):
    __NAME = 'high_to_close_diff'

    def name(self):
        return self.__NAME

    def calculate(self, df: DataFrame):
        return ((df[RawDataColumns.HIGH] - df[RawDataColumns.CLOSE]) / df[RawDataColumns.CLOSE]) * 100
