from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def describe(self) -> None:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def test(self, df: pd.DataFrame):
        pass
