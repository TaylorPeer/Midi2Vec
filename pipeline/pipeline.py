import os
import logging
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from encoding import Encoder
from evaluation import Evaluator, Metrics
from sequence_learning.sequence_learner import SequenceLearner


class Pipeline:
    """
    End-to-end encapsulation of encoder training, sequence learning and evaluation.
    """

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None):
        self._logger = logging.getLogger(__name__)
        self._params = params
        self._encoder = None
        self._data_loader = data_loader
        self._sequence_learner = None
        self._training_docs = training_docs
        self._test_docs = test_docs
        self._k_folds = 0
        self._optimizer = None

        # Encoder training documents cache
        self._encoder_docs = {}

        # Encoder cache
        self._trained_encoders = {}

        # TODO make configurable
        # TODO store doc2vec models here, reload as necessary
        self._temp_dir = "temp"

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
        :return: list of run parameters and evaluation metrics.
        """

        # TODO try/catch to ensure proper shutdown even if error encountered

        params = self._get_params_for_run()
        result_rows = []

        # Check for valid configuration
        if self._test_docs is None and self._k_folds == 0:
            self._logger.error("Explicit test set or number of cross-validation folds must be specified.")
            metrics = Metrics()
            result_row = {**params, **metrics.get_scores_as_dict()}
            result_rows.append(result_row)
            return result_rows

        # Continue while there are configured parameter settings to evaluate
        while params is not None:

            # Get collection of training and test sets for current run
            current_metrics = Metrics()
            data_sets = self._get_training_and_test_sets()
            for data_set_index, (training_docs, test_docs) in enumerate(data_sets):

                # Retrieve an encoder module trained with the specified configuration
                self._encoder = self._get_encoder(params)

                data_set_index += 1  # Only used for user output, so start index at 1

                num_sets = len(data_sets)
                if num_sets > 1:
                    self._logger.info(
                        "Training and evaluating fold " + str(data_set_index) + " of " + str(num_sets) + ".")

                start = time.time()
                self._train_and_eval_seq_learning(params, self._encoder, training_docs, test_docs, current_metrics)
                end = time.time()
                message = "Trained and evaluated fold " + str(data_set_index) + " of sequence model in " + str(
                    end - start) + " seconds."
                self._logger.info(message)

            # Store evaluation metrics of run
            result_row = {**params, **current_metrics.get_scores_as_dict()}
            result_rows.append(result_row)

            # Invoke optimizer callback to report on results of this run
            if self._optimizer is not None:
                self._optimizer.process_run_result(params=params,
                                                   metrics=current_metrics.get_scores_as_dict(),
                                                   encoder=self._encoder,
                                                   sequence_learner=self._sequence_learner)

            # Check if there are additional runs to execute
            if self._optimizer is not None:
                params = self._optimizer.get_next_params()
            else:
                params = None

        # Clear Keras/Tensorflow models
        # (seems to cause a memory leak unless this is called)
        self._sequence_learner.clear_session()

        # Store best model, if configured
        if self._optimizer.is_model_saving_enabled():
            path, name = self._optimizer.get_model_save_path_and_name()
            try:
                self.save(path, name)
            except Exception:
                self._sequence_learner.clear_session()
                self.save(path, name)

        # Clear Keras/Tensorflow models
        # TODO again?
        self._sequence_learner.clear_session()

        return pd.DataFrame(result_rows)

    def save(self, directory, name):
        """
        TODO
        :param directory:
        :param name:
        :return:
        """

        # Get score, round to 2 digits, use as part of save name
        score, (params, encoder, sequence_learner) = self._optimizer.get_best_model()
        score = str(round(score, 2))

        # Create save dir
        full_path = directory + "/" + name + "-" + score
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        with open(full_path + "/" + "settings.json", 'w') as fp:
            json.dump(params, fp)

        encoder.save(full_path + "/" + "encoder")
        sequence_learner.save(full_path + "/" + "seq.h5")
        self._logger.info("Saved model parameters, encoder and sequence learner to: " + str(full_path))

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
            # TODO do not store encoder in memory
            # TODO save to temp directory and re-load
            self._logger.debug("Loading encoder from cache: " + str(encoder_id))
            return self._trained_encoders[encoder_id]
        else:
            self._logger.debug("Training new encoder model: " + str(encoder_id))
            encoder = Encoder(params)
            docs = self._get_docs(encoder, params['doc2vec_docs'])
            encoder.set_documents(docs)
            encoder.train()
            self._trained_encoders[encoder_id] = encoder
            self._logger.debug("Added encoder to cache: " + str(encoder_id))
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
            self._logger.debug("Loading documents from cache: " + path)
            return self._encoder_docs[path]
        else:
            self._logger.debug("Loading documents from disk: " + path)
            docs = encoder.load_documents(path)
            self._encoder_docs[path] = docs
            self._logger.debug("Added documents to cache: " + path)
            return docs

    def _train_and_eval_seq_learning(self, params, encoder, training_docs, test_docs, metrics):
        """
        TODO refactor, split into separate functions (train vs. eval)
        """
        # Load training data with trained encoder
        self._data_loader.set_encoder(encoder)
        self._data_loader.set_params(params)

        # Load training and test data, fitting scaler to training and re-using on test
        training_data = self._data_loader.load_data(training_docs)
        test_data = self._data_loader.load_data(test_docs, fit_scaler=False)
        (x_test, y_test) = test_data

        # Train sequence learning model
        self._sequence_learner = SequenceLearner(params)
        self._sequence_learner.train(training_data)

        # Apply trained model to test set
        predicted = self._sequence_learner.predict(x_test)

        # Evaluate accuracy of model on test set
        # TODO type of evaluator probably depends on data
        # TODO pass from outside (like data_loader)
        evaluator = Evaluator()

        # TODO: is this metric useful?
        # average_error = evaluator.compute_average_error(predicted, y_test)

        # Un-scale predicted and actual values
        scaler = self._data_loader.get_scaler()

        try:
            predicted = scaler.inverse_transform(predicted)
            y_test = scaler.inverse_transform(y_test)
        except ValueError:
            self._logger.error(
                "Unable to un-scale values. \n\tPredicted: \n" + str(predicted) + "\n\tTest: \n" + str(y_test))
            metrics.log_run(precision=0, recall=0, f1=0)
            return

        # Convert actual and predicted vectors to sequence of text values
        predicted_values = encoder.convert_feature_vectors_to_text(predicted)
        actual_values = encoder.convert_feature_vectors_to_text(y_test)

        # Compute accuracy by measuring precision/recall of predicted vs. actual values at every timestamp of evaluation
        (precision, recall, f1) = evaluator.compute_seq_accuracy(predicted_values, actual_values)

        metrics.log_run(precision=precision, recall=recall, f1=f1)
