import logging
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
