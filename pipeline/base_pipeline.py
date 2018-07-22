import os
import time
import logging
import json
import numpy as np
import pandas as pd
from evaluation import Metrics
from sklearn.model_selection import KFold

from encoding import Encoder

from abc import ABC


class BasePipeline(ABC):
    """
    End-to-end encapsulation of encoder training, sequence learning and evaluation.
    """

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None, method=lambda: None):
        self._logger = logging.getLogger(__name__)
        self._params = params
        self._encoder = None
        self._data_loader = data_loader
        self._sequence_learner = None
        self._training_docs = None
        self._test_docs = None
        self._k_folds = 0
        self._optimizer = None
        self._evaluator = None

        self.set_training_docs(training_docs)
        self.set_test_docs(test_docs)

        # Encoder training documents cache
        self._encoder_docs = {}

        # Encoder in-memory cache
        self._trained_encoders = {}

        # Encoder disk-cache directory
        self._encoder_dir = None

        # TODO model saving
        self._save_best_model_path = None
        self._save_best_model_name = None
        self._best_model = None

        # TODO
        self._train_and_evaluate = method

    def set_encoder_cache_dir(self, encoder_cache_dir):
        self._encoder_dir = encoder_cache_dir

    def set_data_loader(self, data_loader):
        self._data_loader = data_loader

    def set_training_docs(self, training_docs):
        # TODO: _collect_files_to_load
        self._training_docs = training_docs

    def set_test_docs(self, test_docs):
        self._test_docs = test_docs

    def set_k_fold_cross_eval(self, k=0):  # TODO rename
        self._k_folds = k

    def set_evaluator(self, evaluator):
        self._evaluator = evaluator

    def get_evaluator(self):
        return self._evaluator

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def _get_best_model(self):
        return self._best_model

    def _set_best_model(self, score, best_model):
        self._best_model = (score, best_model)

    def save_best_model(self, save_path, model_name):
        self._save_best_model_path = save_path
        self._save_best_model_name = model_name

    def _is_model_saving_enabled(self):
        if self._save_best_model_path is not None and self._save_best_model_name is not None:
            return True
        return False

    def _get_model_save_path_and_name(self):
        return self._save_best_model_path, self._save_best_model_name

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
            data_sets = self._get_training_and_test_sets()
            for set_index, (training_docs, test_docs) in enumerate(data_sets):

                # Retrieve an encoder module trained with the specified configuration
                self._encoder = self._get_encoder(params)

                set_index += 1  # Only used for user output, so start index at 1

                num_sets = len(data_sets)
                if num_sets > 1:
                    self._logger.info("Training and evaluating fold {} of {}.".format(set_index, num_sets))

                start = time.time()
                self._train_and_evaluate(params, self._encoder, training_docs, test_docs)
                runtime = time.time() - start
                self._logger.info(
                    "Trained and evaluated fold {} of sequence model in {} seconds.".format(set_index, runtime))

                # Combine run parameters with evaluation results and store
                result_row = {**params, **self._evaluator.get_score_as_dict()}
                result_rows.append(result_row)

                # Check if model should be saved
                if self._is_model_saving_enabled():

                    operator = self._evaluator.get_operator()
                    current_score = self._evaluator.get_score()

                    best_model = self._get_best_model()
                    if best_model is not None:
                        (best_metric, _) = best_model
                        if not operator(best_metric, current_score):
                            self._set_best_model(current_score, (params, self._encoder, self._sequence_learner))
                    else:
                        # New model is the best one if no previous existed
                        self._set_best_model(current_score, (params, self._encoder, self._sequence_learner))

                # Invoke optimizer callback to report on results of this run
                if self._optimizer is not None:
                    self._optimizer.process_run_result(params=params,
                                                       score=self._evaluator.get_score_as_dict(),
                                                       encoder=self._encoder,
                                                       sequence_learner=self._sequence_learner)

            # Check if there are additional runs to execute
            if self._optimizer is not None:
                params = self._optimizer.get_next_params()
            else:
                params = None

        # Store best model, if configured
        if self._is_model_saving_enabled():
            path, name = self._get_model_save_path_and_name()
            try:
                self.save(path, name)
            except Exception:
                self._logger.error("Failed to save model, clearing Keras session and trying again.")
                self._sequence_learner.clear_session()
                self.save(path, name)

        # Clear Keras/Tensorflow models # TODO why a second time?
        if self._sequence_learner is not None:
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
        score, (params, encoder, sequence_learner) = self._get_best_model()
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
        # TODO: refactor method

        # Check if encoder was already trained with these parameters
        encoder_id = Encoder.generate_id(params)
        self._logger.debug("Retrieving encoder model: " + str(encoder_id))

        # Check if matching encoder is in memory
        if encoder_id in self._trained_encoders:
            self._logger.debug("Loading encoder from in-memory cache: " + str(encoder_id))
            return self._trained_encoders[encoder_id]
        else:
            # Check if matching encoder on disk
            prev_model = None
            if self._encoder_dir is not None:
                prev_model = Encoder.load_if_exists(self._encoder_dir, encoder_id)

            if prev_model is not None:
                self._logger.debug("Loaded encoder from disk-cache: " + str(encoder_id))
                encoder = Encoder(params)
                docs = self._get_docs(encoder, params['doc2vec_docs'])
                encoder.set_documents(docs)
                encoder.set_model(prev_model)
                self._trained_encoders[encoder_id] = encoder
                return encoder
            else:
                self._logger.debug("Training new encoder model: " + str(encoder_id))
                encoder = Encoder(params)
                docs = self._get_docs(encoder, params['doc2vec_docs'])
                encoder.set_documents(docs)
                encoder.train()
                self._trained_encoders[encoder_id] = encoder
                self._logger.debug("Added encoder to cache: " + str(encoder_id))

                # Save encoder
                if self._encoder_dir is not None:
                    encoder.save(self._encoder_dir + "/" + encoder_id)
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
