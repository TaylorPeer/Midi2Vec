import operator
from .base_evaluator import BaseEvaluator


class LossEvaluator(BaseEvaluator):

    def __init__(self):
        self._losses = []

    def reset(self):
        self._losses = []

    def log_run(self, y_test, predictions, losses):
        average_loss = sum(losses) / float(len(losses))
        self._losses.append(average_loss)

    def get_score(self):
        loss_score = sum(self._losses) / float(len(self._losses))
        return loss_score

    def get_score_as_dict(self):
        return {"average_loss": self.get_score()}

    @staticmethod
    def get_operator():
        return operator.lt
