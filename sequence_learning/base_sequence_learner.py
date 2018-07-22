import logging
from keras import backend as K
from keras.models import load_model
from keras import optimizers
from keras import callbacks

from abc import ABC, abstractmethod


class BaseSequenceLearner(ABC):
    """
    Generic sequence learning module.
    """

    def __init__(self, params):
        self._logger = logging.getLogger(__name__)
        self._model = None
        self._params = params
        self._loss_history = LossHistory()

    def get_loss_history(self):
        return self._loss_history.get_losses()

    @abstractmethod
    def train(self, training_data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    def save(self, save_to_path):
        """
        Saves the (trained) Keras model.
        :param save_to_path: the path where the model should be stored.
        :return: None.
        """
        if self._model is None:
            self._logger.error("SequenceLearner could not be saved because it is null.")
        else:
            self._model.save(save_to_path)
            self._logger.info("SequenceLearner saved to: " + str(save_to_path))

    def load(self, path_to_stored_model):
        """
        Loads a (trained) Keras model from disk.
        :param path_to_stored_model: the path to load from.
        :return: None.
        """
        self._model = load_model(path_to_stored_model)
        self._logger.info("SequenceLearner loaded from: " + str(path_to_stored_model))

    @staticmethod
    def clear_session():
        """
        Clears session of intermediary variables from old models.
        :return: None.
        """
        K.clear_session()

    @staticmethod
    def _get_optimizer(name):
        """
        TODO
        :param name:
        :return:
        """
        if name == 'sgd':
            return optimizers.SGD()
        if name == 'rmsprop':
            return optimizers.RMSprop()
        if name == 'adagrad':
            return optimizers.Adagrad()
        if name == 'adadelta':
            return optimizers.Adadelta()
        if name == 'adam':
            return optimizers.Adam()
        if name == 'adamax':
            return optimizers.Adamax()
        if name == 'nadam':
            return optimizers.Nadam()

        # TODO what if none matched...
        print("optimizer string was " + str(name))


class LossHistory(callbacks.Callback):

    def __init__(self):
        self._losses = []

    def on_batch_end(self, batch, logs={}):
        self._losses.append(logs.get('loss'))

    def get_losses(self):
        return self._losses
