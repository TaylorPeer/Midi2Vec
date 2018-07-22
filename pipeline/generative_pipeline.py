from sequence_learning import GenerativeSequenceLearner
from pipeline.base_pipeline import BasePipeline


class GenerativePipeline(BasePipeline):

    def __init__(self, params=None, data_loader=None, training_docs=None, test_docs=None):
        super(GenerativePipeline, self).__init__(params=params,
                                                 data_loader=data_loader,
                                                 training_docs=training_docs,
                                                 test_docs=test_docs,
                                                 method=self._train_and_evaluate)

    def _train_and_evaluate(self, params, encoder, training_docs, test_docs):

        # Load training data with trained encoder
        self._data_loader.set_encoder(encoder)
        self._data_loader.set_params(params)

        # Load training and test data, fitting scaler to training and re-using on test
        training_data = self._data_loader.load_data_as_array(training_docs)
        test_data = self._data_loader.load_data_as_array(test_docs, fit_scaler=False)
        (x_test, y_test) = test_data

        # Train sequence learning model
        self._sequence_learner = GenerativeSequenceLearner(params)
        self._sequence_learner.train(training_data)

        # Apply trained model to test set
        predicted = self._sequence_learner.predict(x_test)

        # Un-scale predicted and actual values
        scaler = self._data_loader.get_scaler()

        try:
            predicted = scaler.inverse_transform(predicted)
            y_test = scaler.inverse_transform(y_test)

            # Convert actual and predicted vectors to sequence of text values
            predicted_values = encoder.convert_feature_vectors_to_text(predicted)
            actual_values = encoder.convert_feature_vectors_to_text(y_test)

            evaluator = self.get_evaluator()
            evaluator.reset()
            losses = self._sequence_learner.get_loss_history()
            evaluator.log_run(actual_values, predicted_values, losses)
        except ValueError:
            self._logger.error(
                "Unable to un-scale values. \n\tPredicted: \n{}\n\tTest: {}\n".format(predicted, y_test))
            return
