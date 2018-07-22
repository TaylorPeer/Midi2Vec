from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from sequence_learning.base_sequence_learner import BaseSequenceLearner, LossHistory


class ClassificationSequenceLearner(BaseSequenceLearner):
    """
    Sequence learning module for classification tasks.
    """

    def __init__(self, params, output_dim):
        super(ClassificationSequenceLearner, self).__init__(params)
        if output_dim is None:
            self._logger.error("output_dim was not set when initializing ClassificationSequenceLearner!")
        self._output_dim = output_dim

    def train(self, training_data):
        """
        Trains a sequence learning model on the given training data.
        :param training_data: the training data to use.
        :return: None.
        """

        (x_train, y_train) = training_data

        # Each feature vector is made up of the encoder vector as well as additionally defined custom features
        num_features = self._params['doc2vec_vector_size'] + len(self._params['nn_features'])

        # Construct model as configured
        self._model = self._build_model(num_features=num_features,
                                        layer_count=self._params['nn_layers'],
                                        num_hidden_neurons=self._params['nn_hidden_neurons'],
                                        output_dim=self._output_dim,
                                        lstm_activation=self._params['nn_lstm_activation_function'],
                                        dense_activation=self._params['nn_dense_activation_function'],
                                        dropout_rate=self._params['nn_dropout'],
                                        loss=self._params['nn_loss'],
                                        optimizer=self._params['nn_optimizer'])

        # Train model
        self._model.fit(x_train, y_train,
                        verbose=1,
                        batch_size=self._params['nn_batch_size'],
                        shuffle=True,
                        epochs=self._params['nn_epochs'],
                        callbacks=[self._loss_history])

    def evaluate(self, x_test, y_test):
        """
        TODO
        :param x_test:
        :param y_test:
        :return:
        """
        return self._model.evaluate(x_test, y_test, verbose=0)

    def predict(self, data):
        """
        Predicts the output for an input series.
        :param data: the input series.
        :return: the predicted output.
        """
        if self._model is None:
            self._logger.error("Could not make prediction because sequence learning model has not yet been trained.")
            return None
        return self._model.predict_classes(data)

    @staticmethod
    def _build_model(num_features, layer_count, num_hidden_neurons, output_dim, lstm_activation, dense_activation,
                     dropout_rate, loss, optimizer):
        """
        Builds a deep neural network with the configured parameters.
        :param num_features: size of each feature vector.
        :param layer_count: number of LSTM layers to use.
        :param num_hidden_neurons: number of hidden neurons per layer.
        :param output_dim: TODO
        :param lstm_activation: activation function for each LSTM layer.
        :param dense_activation: activation function for final (dense) layer.
        :param dropout_rate: dropout rate per layer (TODO make configurable per layer type).
        :return: the configured model.
        """

        # Sequences should be returned for multi-layer models
        return_sequences = True if layer_count > 1 else False

        # Construct sequence learning model
        model = Sequential()
        model.add(LSTM(num_hidden_neurons,
                       activation=lstm_activation,
                       return_sequences=return_sequences,
                       input_shape=(None, num_features)))

        # TODO unsure if dropout should be used here
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

        # Add additional layers as configured
        for layer in list(range(1, layer_count)):
            return_sequences = True if (layer + 1) < layer_count else False
            model.add(LSTM(num_hidden_neurons,
                           activation=lstm_activation,
                           return_sequences=return_sequences,
                           input_shape=(None, num_features)))

            # TODO this dropout rate should be able to be set independently
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        # Add final dense layer for output
        model.add(Dense(output_dim, input_dim=num_hidden_neurons))

        # TODO dropout for final layer (?)
        model.add(Activation(dense_activation))

        # TODO optimizer as parameter
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
