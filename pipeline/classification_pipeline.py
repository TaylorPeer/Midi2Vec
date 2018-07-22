from sequence_learning import ClassificationSequenceLearner
from pipeline.base_pipeline import BasePipeline
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder


class ClassificationPipeline(BasePipeline):

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None):
        super(ClassificationPipeline, self).__init__(params=params,
                                                     data_loader=data_loader,
                                                     training_docs=training_docs,
                                                     test_docs=test_docs,
                                                     method=self._train_and_evaluate)

        self._label_encoder = None

    # TODO remove parameters, set in constructor
    def _train_and_evaluate(self, params, encoder, training_docs, test_docs):

        # Load training data with trained encoder
        self._data_loader.set_encoder(encoder)
        self._data_loader.set_params(params)

        # Load training and test data, fitting scaler to training and re-using on test
        training_data = self._data_loader.load_data_as_array(training_docs, pred_labels=True)
        test_data = self._data_loader.load_data_as_array(test_docs, fit_scaler=False, pred_labels=True)
        (x_training, y_training_strings) = training_data
        (x_test, y_test_strings) = test_data

        # Convert string labels to one-hot encoded vectors
        self._fit_label_encoder(y_training_strings)
        y_training_one_hot = self._encode_labels(y_training_strings)
        training_data = (x_training, y_training_one_hot)

        # Train sequence learning model
        output_dimensions = len(y_training_one_hot[0])
        self._sequence_learner = ClassificationSequenceLearner(params, output_dimensions)
        self._sequence_learner.train(training_data)

        # Apply trained model to test set
        predicted = self._sequence_learner.predict(x_test)

        # Evaluate accuracy of model on test set
        try:
            decoded = self._decode_predictions(predicted)
            losses = self._sequence_learner.get_loss_history()
            evaluator = self.get_evaluator()
            evaluator.log_run(y_test_strings, decoded, losses)
        except ValueError:
            self._logger.error("Unable to un-scale values. \n\tPredicted: {0}".format(predicted))

    def _fit_label_encoder(self, labels):
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(labels.flatten())

    def _encode_labels(self, labels):
        integer_encoded = self._label_encoder.transform(labels)
        encoded = np_utils.to_categorical(integer_encoded)
        return encoded

    def _decode_predictions(self, predictions):
        return self._label_encoder.inverse_transform(predictions)
