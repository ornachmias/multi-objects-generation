from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self, dataset):
        self._dataset = dataset

    @abstractmethod
    def generate(self, count):
        pass

