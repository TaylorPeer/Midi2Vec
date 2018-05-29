import logging
import numpy as np
from sklearn.model_selection import KFold

from encoding.encoder import Encoder
from evaluation.evaluator import Evaluator
from sequence_learning.sequence_learner import SequenceLearner
from evaluation.evaluator import Metrics


class Pipeline:
    """
    End-to-end encapsulation of encoder training, sequence learning and evaluation steps.
    """

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None):
        self._logger = logging.getLogger(__name__)
        self._params = params
        self._data_loader = data_loader
        self._training_docs = training_docs
        self._test_docs = test_docs
        self._k_folds = 0
        self._optimizer = None

        # Encoder training documents cache
        self._encoder_docs = {}

        # Encoder cache
        self._trained_encoders = {}

    def set_data_loader(self, data_loader):
        self._data_loader = data_loader

    def set_training_docs(self, training_docs):
        self._training_docs = training_docs

    def set_test_docs(self, test_docs):
        self._test_docs = test_docs

    def set_k_fold_cross_eval(self, k=0):
        self._k_folds = k

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def run(self):
        """
        Runs the full pipeline as configured.
        :return: evaluation metrics.
        """

        params = self._get_params_for_run()
        metrics = Metrics()

        # Check for valid configuration
        if self._test_docs is None and self._k_folds == 0:
            self._logger.error("Explicit test set or number of cross-validation folds must be specified.")
            return metrics

        # Continue while there are configured parameter settings to evaluate
        while params is not None:

            # Get collection of training and test sets for current run
            data_sets = self._get_training_and_test_sets()
            for (training_docs, test_docs) in data_sets:
                # Retrieve an encoder module trained with the specified configuration
                encoder = self._get_encoder(params)

                self._train_and_eval_seq_learning(params, encoder, training_docs, test_docs, metrics)

            params = None
            if self._optimizer is not None:
                # TODO invoke callback in optimizer, to report back on results of this run
                params = self._optimizer.get_next_params()

        return metrics

    def _get_params_for_run(self):
        """
        Retrieves the set of hyperparameters to use for the next run, either from the pipeline configuration or an
        optional optimizer module.
        :return: hyperparameter set to use.
        """
        if self._optimizer is not None:
            return self._optimizer.get_next_params()
        else:
            return self._params

    def _get_training_and_test_sets(self):
        """
        Returns list of training and test sets to use. This may be a single pair (if configured to use a given test and
        training set), or a number of pairs (if configured to use k-fold cross-validation).
        :return: list of training and test sets.
        """

        if self._test_docs is not None:

            # Check for conflicting configuration
            if self._k_folds > 0:
                self._logger.error("Explicit test set cannot be used in conjunction with cross-validation. K-folds "
                                   "setting ignored.")

            # Use explicit training/test sets:
            return [(self._test_docs, self._test_docs)]

        elif self._k_folds > 0:

            # Convert to scalar for use with scalar indices produced by KFold.split
            scalar = np.array(self._training_docs)

            data_sets = []

            # Use k-fold cross-validation to split training docs into training/test splits
            kf = KFold(n_splits=self._k_folds)
            for train_index, test_index in kf.split(scalar):
                training_docs, test_docs = scalar[train_index].tolist(), scalar[test_index].tolist()
                data_sets.append((training_docs, test_docs))

            return data_sets

    def _get_encoder(self, params):
        """
        Retrieves encoder with given parameters, either from cache (if available) or by training a new model.
        :param params: the encoder parameters.
        :return: the trained encoder model.
        """

        encoder_id = Encoder.generate_id(params)

        # Check if already trained
        if encoder_id in self._trained_encoders:
            self._logger.info("Loading encoder from cache: " + str(encoder_id))
            return self._trained_encoders[encoder_id]
        else:
            self._logger.info("Training new encoder model: " + str(encoder_id))
            encoder = Encoder(params)
            docs = self._get_docs(encoder, params['doc2vec_docs'])
            encoder.set_documents(docs)
            encoder.train()
            self._trained_encoders[encoder_id] = encoder
            self._logger.info("Added encoder to cache: " + str(encoder_id))
            return encoder

    def _get_docs(self, encoder, path):
        """
        Loads the documents used to train an encoder, either from disk or from memory (if already loaded).
        :param encoder: an encoder, used to load the documents.
        :param path: the path to load the documents from.
        :return: the docs
        """
        # Check if already loaded
        if path in self._encoder_docs:
            self._logger.info("Loading documents from cache: " + path)
            return self._encoder_docs[path]
        else:
            self._logger.info("Loading documents from disk: " + path)
            docs = encoder.load_documents(path)
            self._encoder_docs[path] = docs
            self._logger.info("Added documents to cache: " + path)
            return docs

    def _train_and_eval_seq_learning(self, params, encoder, training_docs, test_docs, metrics):
        """
        TODO
        :param params:
        :param encoder:
        :param training_docs:
        :param test_docs:
        :param metrics:
        :return:
        """
        # Load training data with trained encoder
        self._data_loader.set_encoder(encoder)
        self._data_loader.set_params(params)

        # TODO check if this set was already loaded with this encoder
        training_data = self._data_loader.load_data(training_docs)

        # Load test data, re-using scaling used during encoding of training set
        # TODO check if this set was already loaded with this encoder
        test_data = self._data_loader.load_data(test_docs, fit_scaler=False)

        (x_test, y_test) = test_data

        # Train sequence learning model
        model = SequenceLearner.train_model(params, training_data)

        # Apply trained model to test set
        predicted = model.predict(x_test)

        # TODO: this shouldn't be necessary...
        SequenceLearner.clear_session()

        # Evaluate accuracy of model on test set
        # TODO type of evaluator probably depends on data
        # TODO pass from outside (like data_loader)
        evaluator = Evaluator()

        # TODO:
        average_error = evaluator.compute_average_error(predicted, y_test)

        # Un-scale predicted and actual values
        scaler = self._data_loader.get_scaler()
        predicted = scaler.inverse_transform(predicted)
        y_test = scaler.inverse_transform(y_test)

        # Convert predicted vectors to sequence of text values
        predicted_values = encoder.convert_vectors_to_text(predicted)

        # Convert actual vectors to sequence of text values
        actual_values = encoder.convert_vectors_to_text(y_test)

        # Compute accuracy by measuring precision/recall of predicted vs. actual values at every timestamp of evaluation
        (precision, recall, f1) = evaluator.compute_seq_accuracy(predicted_values, actual_values)

        metrics.log_run(precision=precision, recall=recall, f1=f1)
