import logging
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base optimizer module.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._callback = None
        self._save_best_model_path = None
        self._save_best_model_name = None
        self._best_model = None

    @abstractmethod
    def get_next_params(self):
        pass

    @abstractmethod
    def process_run_result(self, params, metrics, encoder, sequence_learner):
        pass

    def set_callback(self, callback):
        self._callback = callback

    def get_callback(self):
        return self._callback

    def get_best_model(self):
        return self._best_model

    def set_best_model(self, score, best_model):
        self._best_model = (score, best_model)

    def save_best_model(self, save_path, model_name):
        self._save_best_model_path = save_path
        self._save_best_model_name = model_name

    def is_model_saving_enabled(self):
        if self._save_best_model_path is not None and self._save_best_model_name is not None:
            return True
        return False

    def get_model_save_path_and_name(self):
        return self._save_best_model_path, self._save_best_model_name
