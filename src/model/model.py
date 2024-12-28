from abc import ABC,abstractmethod

class Model(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def describe(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def train(self):
        pass
