import unittest
import numpy as np
from collections import namedtuple
from evaluation.evaluator import Evaluator


class EvaluatorTests(unittest.TestCase):
    """
    Tests basic evaluation metric computation.
    """

    def __init__(self, *args, **kwargs):
        super(EvaluatorTests, self).__init__(*args, **kwargs)
        self.test_instance = Evaluator()

    def test_row_metrics(self):
        """
        Tests correct computation of row-wise metrics.
        :return: None.
        """
        TestCase = namedtuple('Testcase', 'actual predicted precision recall')

        cases = []
        cases.append(TestCase('percussion_openhat_30', 'percussion_openhat_30', 1, 1))
        cases.append(TestCase('percussion_openhat_30', 'percussion_openhat_30, percussion_rimshot_30', 0.5, 1))
        cases.append(TestCase('percussion_openhat_30, percussion_rimshot_30', 'percussion_openhat_30', 1, 0.5))
        cases.append(TestCase('percussion_rimshot_30', 'percussion_openhat_30', 0, 0))

        for case in cases:
            (precision, recall, fscore) = self.test_instance.compute_row_accuracy(case.predicted, case.actual)
            self.assertTrue(precision == case.precision)
            self.assertTrue(recall == case.recall)

        return

    def test_seq_metrics(self):
        """
        Tests correct computation of sequence-based (average) metrics.
        :return: None.
        """
        TestCase = namedtuple('Testcase', 'actual predicted precision recall')
        cases = []
        cases.append(TestCase(['percussion_openhat_30', 'percussion_rimshot_30'],
                              ['percussion_openhat_30', 'percussion_rimshot_30'], 1, 1))
        cases.append(TestCase(['percussion_openhat_30', 'percussion_openhat_30'],
                              ['percussion_openhat_30, percussion_rimshot_30',
                               'percussion_openhat_30, percussion_rimshot_30'], 0.5, 1))
        cases.append(
            TestCase(['percussion_openhat_30, percussion_rimshot_30', 'percussion_openhat_30, percussion_rimshot_30'],
                     ['percussion_openhat_30', 'percussion_openhat_30'], 1, 0.5))
        cases.append(TestCase(['percussion_rimshot_30', 'percussion_rimshot_30'],
                              ['percussion_openhat_30', 'percussion_openhat_30'], 0, 0))

        for case in cases:
            (precision, recall, fscore) = self.test_instance.compute_seq_accuracy(case.predicted, case.actual)
            self.assertTrue(precision == case.precision)
            self.assertTrue(recall == case.recall)

        return

    def test_error_metrics(self):
        """
        Tests that computation of average error between vectors returns reasonable values.
        :return: None.
        """
        predicted = np.random.rand(10, 2)
        actual = np.random.rand(10, 2)
        error = self.test_instance.compute_average_error(predicted, actual)
        self.assertTrue(error > 0)
        return


def main():
    unittest.main()


if __name__ == '__main__':
    main()
