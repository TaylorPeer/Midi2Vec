from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation


class SequenceLearner:
    """
    TODO
    """

    @staticmethod
    def train_model(params, training_data):
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

        (x_train, y_train) = training_data

        # Build model
        model.fit(x_train, y_train, verbose=0, batch_size=params['nn_batch_size'], epochs=params['nn_epochs'])

        return model
