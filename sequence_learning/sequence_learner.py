import logging
import time
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras import backend as K
from keras.models import load_model
from keras import optimizers


class SequenceLearner:
    """
    Sequence learning object.
    """

    def __init__(self, params):
        self._logger = logging.getLogger(__name__)
        self._model = None
        self._params = params

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
                                        lstm_activation=self._params['nn_lstm_activation_function'],
                                        dense_activation=self._params['nn_dense_activation_function'],
                                        dropout_rate=self._params['nn_dropout'],
                                        loss=self._params['nn_loss'],
                                        optimizer=self._params['nn_optimizer'])

        # Train model
        # TODO "verbose" as configurable parameter (for debuggign)
        self._model.fit(x_train, y_train, verbose=0, batch_size=self._params['nn_batch_size'],
                        epochs=self._params['nn_epochs'])

    def predict(self, data):
        """
        Predicts the output for an input series.
        :param data: the input series.
        :return: the predicted output.
        """
        if self._model is None:
            self._logger.error("Could not make prediction because sequence learning model has not yet been trained.")
            return None
        return self._model.predict(data)

    def generate_sequence(self, seed_df, data_loader, length):
        """
        # TODO move to sequence_generator?
        Generates a new sequence of a given length using a sample dataframe of data as seed values.
        :param seed_df: the Pandas dataframe to use as seed values.
        :param data_loader: the DataLoader used to load the training data.
        :param length: the length of the sequence to generate, in discrete steps.
        :return: the generated sequence (as a Pandas dataframe).
        """

        # TODO make configurable: retain seed sequence as part of generated sequence or not

        if self._model is None:
            self._logger.error("Could not generate sequence because sequence learning model has not yet been trained.")
            return pd.DataFrame()

        scaler = data_loader.get_scaler()
        if scaler is None:
            self._logger.error("DataLoader scaler was null. Was it used to load the training data?")
            return pd.DataFrame()

        # Apply pre-processing to seed Data Frame
        pattern_df_scaled = pd.DataFrame(scaler.transform(seed_df), columns=seed_df.columns)
        (seed_x, _) = data_loader.frame_as_sequential(pattern_df_scaled, self._params['nn_lstm_n_prev'])

        generated_rows = []
        for step in range(length):
            # Predict step
            prediction = self._model.predict(seed_x, verbose=0)
            prediction = prediction[0]

            # Create dictionary out of predicted vector
            predicted_vector = {}
            for index, column in enumerate(pattern_df_scaled):
                predicted_vector[column] = prediction[index]

            # Append prediction to seed sequence and remove oldest step in seed sequence
            pattern_df_scaled = pd.concat([pattern_df_scaled, pd.DataFrame(predicted_vector, index=[0])])
            pattern_df_scaled = pattern_df_scaled.iloc[1:]

            # Translate predicted vector into textual representation:
            predicted_df = pd.DataFrame(prediction.reshape(-1, len(prediction)))
            scaler = data_loader.get_scaler()
            predicted_df = pd.DataFrame(scaler.inverse_transform(predicted_df), columns=predicted_df.columns)
            predicted_vector = np.array(predicted_df.loc[0])

            # Create intermediary dictionary to hold generated row (features and textual values)
            row_dict = {}
            features = self._params['nn_features']
            predicted_features = predicted_vector[:len(features)]
            for index, _ in enumerate(features):
                row_dict[features[index]] = predicted_features[index]

            # Lookup most similar vector encountered in training set
            encoder = data_loader.get_encoder()
            predicted_values = encoder.convert_feature_vector_to_text(predicted_vector)
            row_dict['notes'] = predicted_values  # TODO remove MIDI-specific values ('notes') from this class

            # Record predicted vector
            generated_rows.append(row_dict)

            # Re-run preprocessing on updated seed data frame
            (seed_x, _) = data_loader.frame_as_sequential(pattern_df_scaled, self._params['nn_lstm_n_prev'])

        return pd.DataFrame(generated_rows)

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def _build_model(num_features, layer_count, num_hidden_neurons, lstm_activation, dense_activation,
                     dropout_rate, loss, optimizer):
        """
        Builds a deep neural network with the configured parameters.
        :param num_features: size of each feature vector.
        :param layer_count: number of LSTM layers to use.
        :param num_hidden_neurons: number of hidden neurons per layer.
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
        model.add(Dense(num_features, input_dim=num_hidden_neurons))
        # TODO dropout for final layer (?)
        model.add(Activation(dense_activation))

        model.compile(loss=loss, optimizer=SequenceLearner._get_optimizer(optimizer))
        return model

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
