import logging
import itertools
from random import shuffle
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base optimizer module.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_next_params(self):
        pass


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

        # Randomize order
        shuffle(self._combinations)

    def get_next_params(self):
        if len(self._combinations) > 0:
            params = dict(self._combinations.pop(0))
            self.logger.info("Returning next set of hyperparameters: " + str(params))
            return params
        return None
