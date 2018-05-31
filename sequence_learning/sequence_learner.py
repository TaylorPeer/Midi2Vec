import logging
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation
from keras import backend as K


class SequenceLearner:
    """
    Sequence learning module.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._model = None
        self._params = None

    def train(self, params, training_data):
        """
        Trains a sequence learning model on the given training data.
        :param params: TODO replace params with individual named parameters
        :param training_data: the training data to use.
        :return: None.
        """

        self._params = params
        (x_train, y_train) = training_data

        in_out_neurons = params['doc2vec_vector_size'] + len(params['nn_features'])
        hidden_neurons = params['nn_hidden_neurons']
        layers = params['nn_layers']
        return_sequences = True if layers > 1 else False

        model = Sequential()
        model.add(
            LSTM(hidden_neurons, activation=params['nn_lstm_activation_function'], return_sequences=return_sequences,
                 input_shape=(None, in_out_neurons)))
        model.add(Dropout(params['nn_dropout']))
        for layer in list(range(1, layers)):
            return_sequences = True if (layer + 1) < layers else False
            model.add(LSTM(hidden_neurons, activation=params['nn_lstm_activation_function'],
                           return_sequences=return_sequences, input_shape=(None, in_out_neurons)))
            model.add(Dropout(params['nn_dropout']))
        model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
        model.add(Activation(params['nn_dense_activation_function']))
        model.compile(loss="mean_squared_error", optimizer="rmsprop")  # TODO optimizer/loss as parameters

        # Build model
        # TODO "verbose" as parameter
        model.fit(x_train, y_train, verbose=0, batch_size=params['nn_batch_size'], epochs=params['nn_epochs'])
        self._model = model

    def predict(self, data):
        """
        TODO
        :param data:
        :return:
        """
        if self._model is None:
            self._logger.error("Could not make prediction because sequence learning model has not yet been trained.")
            return None
        return self._model.predict(data)

    def generate_sequence(self, seed_df, data_loader, length):

        # TODO make configurable: retain seed sequence as part of generated sequence or not

        if self._model is None:
            self._logger.error("Could not generate sequence because sequence learning model has not yet been trained.")
            return None

        scaler = data_loader.get_scaler()
        if scaler is None:
            print("what do?")
            return

        pattern_df_scaled = pd.DataFrame(scaler.transform(seed_df), columns=seed_df.columns)
        (seed_x, _) = data_loader.frame_as_sequential(pattern_df_scaled, self._params['nn_lstm_n_prev'])

        predicted_df_rows = []

        for step in range(length):
            # Predict step
            prediction = self._model.predict(seed_x, verbose=0)
            prediction = prediction[0]

            # Create dictionary out of predicted vector
            predicted_vector = {}
            for index, column in enumerate(pattern_df_scaled):
                predicted_vector[column] = prediction[index]

            # Record predicted vector
            predicted_df_rows.append(predicted_vector)

            # Append prediction to seed sequence and remove oldest step in seed sequence
            pattern_df_scaled = pd.concat([pattern_df_scaled, pd.DataFrame(predicted_vector, index=[0])])
            pattern_df_scaled = pattern_df_scaled.iloc[1:]

            # Translate predicted vector into textual representation:
            predicted_df = pd.DataFrame(prediction.reshape(-1, len(prediction)))
            scaler = data_loader.get_scaler()
            predicted_df = pd.DataFrame(scaler.inverse_transform(predicted_df), columns=predicted_df.columns)
            predicted_vector = np.array(predicted_df.loc[0])

            # Lookup most similar vector encountered in training set
            encoder = data_loader.get_encoder()
            values = encoder.convert_vectors_to_text([predicted_vector])
            print(values)

            (seed_x, _) = data_loader.frame_as_sequential(pattern_df_scaled, self._params['nn_lstm_n_prev'])

        return pd.DataFrame(predicted_df_rows)

    @staticmethod
    def clear_session():
        K.clear_session()
