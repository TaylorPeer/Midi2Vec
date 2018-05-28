import logging
import numpy as np
from sklearn.model_selection import KFold

from encoding.encoder import Encoder
from evaluation.evaluator import Evaluator
from sequence_learning.sequence_learner import SequenceLearner
from evaluation.evaluator import Metrics


class Pipeline:
    """
    TODO...
    """

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None):
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.data_loader = data_loader
        self.training_docs = training_docs
        self.test_docs = test_docs
        self.k_folds = 0

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_training_docs(self, training_docs):
        self.training_docs = training_docs

    def set_test_docs(self, test_docs):
        self.test_docs = test_docs

    def set_k_fold_cross_eval(self, k=0):
        self.k_folds = k

    def set_optimizer(self, optimizer):
        pass

    def run(self):

        # Train encoder
        encoder = Encoder(self.params)
        docs = encoder.load_documents(self.params['doc2vec_docs'])
        encoder.set_documents(docs)
        encoder.train()

        metrics = Metrics()
        runs = []

        if self.test_docs is not None:

            # Check for conflicting configuration
            if self.k_folds > 0:
                self.logger.error("Explicit test set cannot be used in conjunction with cross-validation. K-folds "
                                  "setting ignored.")

            # Use explicit training/test sets:
            runs.append((self.test_docs, self.test_docs))

        elif self.k_folds > 0:

            # Convert to scalar for use with scalar indices produced by KFold.split
            scalar = np.array(self.training_docs)

            # Use k-fold cross-validation to split training docs into training/test splits
            kf = KFold(n_splits=self.k_folds)
            for train_index, test_index in kf.split(scalar):
                training_docs, test_docs = scalar[train_index].tolist(), scalar[test_index].tolist()
                runs.append((training_docs, test_docs))

        elif self.test_docs is None and self.k_folds == 0:
            self.logger.error("Explicit test set or number of cross-validation folds must be specified.")
            return metrics

        for index, run in enumerate(runs):
            training_docs, test_docs = runs[index]

            # Load training data with trained encoder
            self.data_loader.set_encoder(encoder)
            training_data = self.data_loader.load_data(training_docs)

            # Load test data, re-using scaling used during encoding of training set
            test_data = self.data_loader.load_data(test_docs, fit_scaler=False)

            (x_test, y_test) = test_data

            # Train sequence learning model
            model = SequenceLearner.train_model(self.params, training_data)

            # Apply trained model to test set
            predicted = model.predict(x_test)

            # TODO:
            SequenceLearner.clear_session()

            # Evaluate accuracy of model on test set
            # TODO type of evaluator probably depends on data
            # TODO pass from outside (like data_loader)
            evaluator = Evaluator()

            # TODO:
            average_error = evaluator.compute_average_error(predicted, y_test)

            # Un-scale predicted and actual values
            scaler = self.data_loader.get_scaler()
            predicted = scaler.inverse_transform(predicted)
            y_test = scaler.inverse_transform(y_test)

            # Convert predicted vectors to note sequence
            predicted_notes = encoder.convert_vectors_to_text(predicted)

            # Convert actual vectors to note sequence
            actual_notes = encoder.convert_vectors_to_text(y_test)

            # Compute accuracy by measuring precision/recall of predicted vs. actual notes at every timestamp of evaluation
            (precision, recall, f1) = evaluator.compute_seq_accuracy(predicted_notes, actual_notes)

            metrics.log_run(precision=precision, recall=recall, f1=f1)

        return metrics
