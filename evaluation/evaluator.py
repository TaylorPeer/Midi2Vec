import math
import logging

from scipy.spatial import distance


class Metrics:

    def __init__(self):
        self._total_precision = 0
        self._total_recall = 0
        self._total_f1 = 0
        self._runs = 0

    def log_run(self, precision=None, recall=None, f1=None):
        self._runs += 1
        if precision is not None:
            self._total_precision += precision
        if recall is not None:
            self._total_recall += recall
        if f1 is not None:
            self._total_f1 += f1

    def get_precision(self):
        return self._total_precision / self._runs

    def get_recall(self):
        return self._total_recall / self._runs

    def get_f1(self):
        return self._total_f1 / self._runs


class Evaluator:
    """
    Class encapsulating methods for evaluating predictions.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compute_average_error(self, predicted, actual):
        """
        Computes the average error between predicted and actual (test) vectors.
        :param predicted: the predicted vectors.
        :param actual: the actual (test) vectors.
        :return: average difference (error) between predicted and actual vectors.
        """
        distances = list(self.compute_errors(predicted, actual))
        average = sum(distances) / float(len(distances))
        return average

    def compute_errors(self, predicted, actual):
        """
        Computes the error as the difference/distance between predicted and actual (test) vectors.
        :param predicted: the predicted vector.
        :param actual: the actual (test) vector.
        :return: the difference/distance (error) between the predicted and actual vector.
        """
        for index in range(0, len(predicted)):
            cos_distance = distance.cosine(predicted[index], actual[index])

            if math.isnan(cos_distance):
                # Occurs if predicted or actual was an array of 0's
                # This probably only happens if doc2vec model is undertrained, i.e. some weights were still 0
                self.logger.warning("Cosine distance between predicted " + str(predicted[index]) + " and test " + str(
                    actual[index]) + " vectors was NaN")
                cos_distance = 1

            yield cos_distance

    def compute_seq_accuracy(self, predicted, actual):
        """
        Computes various accuracy metrics with respect to predicted and actual (test) vectors.
        :param predicted: the predicted vectors.
        :param actual: the actual vectors.
        :return: a tuple of average accuracy metrics (precision, recall, and F1 score)
        """

        # Metrics to compute
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0

        # Compute metrics for each prediction step
        for index, predicted_row in enumerate(predicted):
            (precision_row, recall_row, f1_row) = self.compute_row_accuracy(predicted_row, actual[index])
            precision_sum += precision_row
            recall_sum += recall_row
            f1_sum += f1_row

        # Adjust metrics by number of predictions
        precision = precision_sum / float(len(predicted))
        recall = recall_sum / float(len(predicted))
        f1 = f1_sum / float(len(predicted))

        return precision, recall, f1

    @staticmethod
    def compute_row_accuracy(predicted_row, actual_row):
        """
        Computes various accuracy metrics with respect to a predicted and an actual (test) vector.
        :param predicted_row: the row of predicted values.
        :param actual_row: the row of actual (test) values.
        :return: a tuple of accuracy metrics (precision, recall, and F1 score)
        """

        # Retrieve individual predicted values
        predicted_values = predicted_row.split(",")
        actual_values = actual_row.split(",")

        # How many predicted values actually occurred:
        # TODO account for duplicate values within single row
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

        return precision, recall, f1
