from abc import ABC, abstractmethod

from pandas import DataFrame


class Feature(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def calculate(self, df: DataFrame):
        pass
