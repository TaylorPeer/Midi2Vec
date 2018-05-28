import logging

from encoding.encoder import Encoder
from evaluation.evaluator import Evaluator
from sequence_learning.sequence_learner import SequenceLearner
from evaluation.evaluator import Metrics


class Pipeline:

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

    def set_k_fold_cross_eval(self, k):
        self.k_folds = k

    def set_optimizer(self, optimizer):
        pass

    def run(self):

        # Train encoder
        encoder = Encoder(self.params)
        docs = encoder.load_documents(self.params['doc2vec_docs'])
        encoder.set_documents(docs)
        encoder.train()

        # Load training data with trained encoder
        self.data_loader.set_encoder(encoder)
        training_data = self.data_loader.load_data(self.training_docs)

        # Load test data, re-using scaling used during encoding of training set
        test_data = self.data_loader.load_data(self.test_docs, fit_scaler=False)
        (x_test, y_test) = test_data

        # Train sequence learning model
        model = SequenceLearner.train_model(self.params, training_data)

        # Apply trained model to test set
        predicted = model.predict(x_test)

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

        return Metrics(precision=precision, recall=recall, f1=f1)
