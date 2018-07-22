import itertools
from random import shuffle
from .optimizers import Optimizer


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

        self._logger.info("Brute-force evaluation of " + str(len(self._combinations)) + " hyperparameter combinations.")

        # Randomize order
        shuffle(self._combinations)

    def get_next_params(self):
        if len(self._combinations) > 0:
            params = dict(self._combinations.pop(0))
            self._logger.debug("Returning next set of hyperparameters: " + str(params))
            return params
        return None

    def process_run_result(self, params, score, encoder, sequence_learner):
        """
        Processes the results of a single run of the pipeline.
        :param params: the hyperparameter settings of the run.
        :param score: the evaluation score of the run.
        :param encoder: the encoder used during the run.
        :param sequence_learner: the sequence learner used during the run
        :return: None
        """

        callback = self.get_callback()
        if callback is not None:
            callback(params=params, score=score, abort=self._abort)

    def _abort(self):
        """
        Aborts evaluation by clearing the remaining hyperparameter combinations left to evaluate.
        :return: None.
        """
        self._logger.info("Aborting brute-force evaluation.")
        self._combinations = []
