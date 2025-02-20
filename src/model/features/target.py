from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from pandas import DataFrame

from data.raw_data_columns import DataColumns
from model.features.feature import Feature
from utils.log import get_logger


class TargetLabel(Enum):
    DOWN = 0
    UP = 1
    NEUTRAL = 2


class TargetLabelingPolicy(ABC):

    @abstractmethod
    def label(self, df: DataFrame):
        pass

    @abstractmethod
    def threshold_up(self, df: DataFrame):
        pass

    @abstractmethod
    def threshold_down(self, df: DataFrame):
        pass


class Target(Feature):
    def __init__(self, labeling_policy: TargetLabelingPolicy):
        super().__init__(DataColumns.TARGET)
        self.__labeling_policy = labeling_policy

    def _calculate(self, input_df: DataFrame):
        df = input_df.copy()
        df[DataColumns.CLOSE_PERCENT_CHANGE] = np.round(df[DataColumns.CLOSE].pct_change() * 100, 5)
        df[DataColumns.TARGET] = self.__labeling_policy.label(df)
        df[DataColumns.TARGET] = df[DataColumns.TARGET].shift(-1)
        return df[DataColumns.TARGET]

    def threshold_up(self, df: DataFrame):
        return self.__labeling_policy.threshold_up(df)

    def threshold_down(self, df: DataFrame):
        return self.__labeling_policy.threshold_down(df)


class AtrLabelingPolicy(TargetLabelingPolicy):
    def __init__(self, atr_window):
        self.__atr_window = atr_window

    def threshold_up(self, df: DataFrame):
        high_to_low_diff = df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change() * 100
        high_to_previous_low_diff = np.abs(df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change().shift() * 100)
        low_to_previous_close = np.abs(df[DataColumns.LOW].pct_change() * 100 - df[DataColumns.CLOSE].pct_change().shift() * 100)
        df['TrueRange'] = np.maximum(high_to_low_diff, high_to_previous_low_diff, low_to_previous_close)
        df['ATR'] = df['TrueRange'].rolling(window=self.__atr_window).mean()
        return df['ATR']

    def threshold_down(self, df: DataFrame):
        high_to_low_diff = df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change() * 100
        high_to_previous_low_diff = np.abs(df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change().shift() * 100)
        low_to_previous_close = np.abs(df[DataColumns.LOW].pct_change() * 100 - df[DataColumns.CLOSE].pct_change().shift() * 100)
        df['TrueRange'] = np.maximum(high_to_low_diff, high_to_previous_low_diff, low_to_previous_close)
        df['ATR'] = df['TrueRange'].rolling(window=self.__atr_window).mean()
        return -df['ATR']

    def label(self, df: DataFrame):
        high_to_low_diff = df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change() * 100
        high_to_previous_low_diff = np.abs(df[DataColumns.HIGH].pct_change() * 100 - df[DataColumns.LOW].pct_change().shift() * 100)
        low_to_previous_close = np.abs(df[DataColumns.LOW].pct_change() * 100 - df[DataColumns.CLOSE].pct_change().shift() * 100)
        df['TrueRange'] = np.maximum(high_to_low_diff, high_to_previous_low_diff, low_to_previous_close)
        df['ATR'] = df['TrueRange'].rolling(window=self.__atr_window).mean()
        return df.apply(self.__label_with_atr, axis=1)

    def __label_with_atr(self, row):
        if row[DataColumns.CLOSE_PERCENT_CHANGE] > row['ATR']:
            return TargetLabel.UP.value
        elif row[DataColumns.CLOSE_PERCENT_CHANGE] < -row['ATR']:
            return TargetLabel.DOWN.value
        else:
            return TargetLabel.NEUTRAL.value


class PercentileLabelingPolicy(TargetLabelingPolicy):
    def __init__(self, negative_percentile: int, positive_percentile: int):
        self.__negative_percentile = negative_percentile
        self.__positive_percentile = positive_percentile
        self.calculated_threshold_up = None
        self.calculated_threshold_down = None

    def threshold_up(self, df: DataFrame):
        return df.apply(lambda _: self.calculated_threshold_up, axis=1)

    def threshold_down(self, df: DataFrame):
        return df.apply(lambda x: self.calculated_threshold_down, axis=1)

    def label(self, df: DataFrame):
        if self.calculated_threshold_up is None:
            positive_changes = df[df[DataColumns.CLOSE_PERCENT_CHANGE] > 0][DataColumns.CLOSE_PERCENT_CHANGE]
            negative_changes = df[df[DataColumns.CLOSE_PERCENT_CHANGE] < 0][DataColumns.CLOSE_PERCENT_CHANGE]

            positive_percentile = round(self.__negative_percentile / 100, 2)
            negative_percentile = round(self.__positive_percentile / 100, 2)

            positive_percentiles = positive_changes.describe(percentiles=[negative_percentile])
            negative_percentiles = negative_changes.describe(percentiles=[positive_percentile])

            self.calculated_threshold_up = positive_percentiles.loc[f'{self.__positive_percentile}%']
            self.calculated_threshold_down = negative_percentiles.loc[f'{self.__negative_percentile}%']
            get_logger('target').info(f'Percentiles DOWN: {self.__negative_percentile}, UP: {self.__positive_percentile}. Calc thresholds: {self.calculated_threshold_up}, {self.calculated_threshold_down}')
        return df[DataColumns.CLOSE_PERCENT_CHANGE].apply(self.__calculate_labels)

    def __calculate_labels(self, row):
        if row < self.calculated_threshold_down:
            return TargetLabel.DOWN.value
        elif self.calculated_threshold_down <= row <= self.calculated_threshold_up:
            return TargetLabel.NEUTRAL.value
        else:
            return TargetLabel.UP.value
