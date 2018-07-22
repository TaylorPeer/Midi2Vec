import operator
from .base_evaluator import BaseEvaluator


class F1Evaluator(BaseEvaluator):

    def __init__(self):
        self._f1_scores = []

    def reset(self):
        self._f1_scores = []

    def log_run(self, actual_notes, predicted_notes, losses):
        f1 = self._compute_average_f1(predicted_notes, actual_notes)
        self._f1_scores.append(f1)

    def get_score(self):
        return sum(self._f1_scores) / float(len(self._f1_scores))

    def get_score_as_dict(self):
        return {"f1": self.get_score()}

    @staticmethod
    def get_operator():
        return operator.gt

    @staticmethod
    def _compute_average_f1(predicted, actual):
        """
        TODO
        :param predicted: the predicted vectors.
        :param actual: the actual vectors.
        :return: F1 score.
        """

        # Metrics to compute
        f1_sum = 0

        # Compute metrics for each prediction step
        for index, predicted_row in enumerate(predicted):
            f1_sum += F1Evaluator._compute_f1(predicted_row, actual[index])

        # Adjust metrics by number of predictions
        f1 = f1_sum / float(len(predicted))

        return f1

    @staticmethod
    def _compute_f1(predicted_row, actual_row):
        """
        TODO
        :param predicted_row: the row of predicted values.
        :param actual_row: the row of actual (test) values.
        :return: F1 score
        """

        # Retrieve individual predicted values
        predicted_values = predicted_row.split(",")
        actual_values = actual_row.split(",")

        # How many predicted values actually occurred:
        # TODO account for possible duplicate values within single row
        precision = 0
        for predicted_value in predicted_values:
            if predicted_value in actual_values:
                precision += 1

        # How many actual values were predicted:
        recall = 0
        for actual_value in actual_values:
            if actual_value in predicted_values:
                recall += 1

        # Adjust precision and recall scores by length of vectors
        precision = precision / float(len(predicted_values))
        recall = recall / float(len(actual_values))

        # Compute F1 score
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall > 0) else 0

        return f1
