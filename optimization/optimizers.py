import logging
import itertools
from random import shuffle
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base optimizer module.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._callback = None

    @abstractmethod
    def get_next_params(self):
        pass

    @abstractmethod
    def process_run_result(self, params, metrics):
        pass

    def set_callback(self, callback):
        self._callback = callback

    def get_callback(self):
        return self._callback


class BruteForce(Optimizer):
    """
    Brute force optimizer that computes all permutations of a given list of hyperparameters, to be used in conjunction
    with a Pipeline model.
    """

    def __init__(self, params):
        super(BruteForce, self).__init__()

        # Create list of all possible parameter value combinations
        values = [[(key, value) for value in values] for (key, values) in sorted(params.items())]
        self._combinations = list(itertools.product(*values))

        self._logger.info("BruteForce evaluation of " + str(len(self._combinations)) + " hyperparameter combinations.")

        # Randomize order
        shuffle(self._combinations)

    def get_next_params(self):
        if len(self._combinations) > 0:
            params = dict(self._combinations.pop(0))
            self._logger.debug("Returning next set of hyperparameters: " + str(params))
            return params
        return None

    def process_run_result(self, params, metrics):
        """
        Processes the results of a single run of the pipeline.
        :param params: the hyperparameter settings of the run.
        :param metrics: the evaluation metrics of the run.
        :return: None
        """
        callback = self.get_callback()
        if callback is not None:
            callback(params=params, metrics=metrics)
