from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def log_run(self, y_test, predictions, losses):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def get_score_as_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def get_operator():
        pass
