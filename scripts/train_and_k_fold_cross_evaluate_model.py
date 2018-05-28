import sys
import logging
import numpy as np
from sklearn.model_selection import KFold

from encoding.encoder import Encoder
from evaluation.evaluator import Evaluator
from sequence_learning.sequence_learner import SequenceLearner
from data_loading.midi_data_loader import MidiDataLoader

from midi_to_dataframe.note_mapper import NoteMapper

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def main():
    # Load documents for encoder training
    path_to_documents = "/Users/taylorpeer/Projects/se-project/midi-embeddings/data/full_1_measure_100k.txt"
    params = get_model_parameters(path_to_documents)

    # Train encoder
    encoder = Encoder(params)
    docs = encoder.load_documents(path_to_documents)
    encoder.set_documents(docs)
    encoder.train()

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Define training documents for sequence learning
    training_docs = []
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/084 Breakthru.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/086 Clouds.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/089 Get Out.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/089 Wrong.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/090 Deceive.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/090 New York.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/090 Radio.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/093 Pretender.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/093 Right Won.mid")
    training_docs.append(
        "/Users/taylorpeer/Projects/se-project/midi-embeddings/backend/scripts/resources/breakbeats/094 Run.mid")

    total_error = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    scalar_array = np.array(training_docs)

    # k-fold on training_docs...
    k = 5
    kf = KFold(n_splits=k)
    for fold, (train_index, test_index) in enumerate(kf.split(scalar_array)):
        training_docs, evaluation_docs = scalar_array[train_index].tolist(), scalar_array[test_index].tolist()

        # Load training MIDI files using MidiDataLoader
        data_loader = MidiDataLoader(params, note_mapper, encoder)
        training_data = data_loader.load_data(training_docs)

        # Set fit_scaler= False to re-use scaler from training set
        test_data = data_loader.load_data(evaluation_docs, fit_scaler=False)
        (x_test, y_test) = test_data

        # Train sequence learning model
        model = SequenceLearner.train_model(params, training_data)

        # Apply trained model to test set
        predicted = model.predict(x_test)

        # Evaluate accuracy of model on test set
        evaluator = Evaluator()
        average_error = evaluator.compute_average_error(predicted, y_test)

        # Un-scale predicted and actual values
        scaler = data_loader.get_scaler()
        predicted = scaler.inverse_transform(predicted)
        y_test = scaler.inverse_transform(y_test)

        # Convert predicted vectors to note sequence
        predicted_notes = encoder.convert_vectors_to_text(predicted)

        # Convert actual vectors to note sequence
        actual_notes = encoder.convert_vectors_to_text(y_test)

        # Compute accuracy by measuring precision/recall of predicted vs. actual notes at every timestamp of evaluation
        (precision, recall, f1) = evaluator.compute_seq_accuracy(predicted_notes, actual_notes)

        print("Accuracy for Fold: " + str(fold))
        print("- precision: " + str(precision))
        print("- recall: " + str(recall))
        print("- f1: " + str(f1))
        print("- average error: " + str(average_error))

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_error += average_error

    # Adjust total metric values by number of folds
    average_error = total_error / k
    average_precision = total_precision / k
    average_recall = total_recall / k
    average_f1 = total_f1 / k

    # Remove doc2vec_docs params setting, since otherwise params can't be printed
    params = dict((key, value) for key, value in params.items() if key != 'doc2vec_docs')

    print(str(params))
    print("- precision: " + str(average_precision))
    print("- recall: " + str(average_recall))
    print("- f1: " + str(average_f1))
    print("- average error: " + str(average_error))
    print("---")


def get_model_parameters(docs):
    params = {}

    # Doc2vec values
    params['doc2vec_docs'] = docs
    params['doc2vec_dm'] = 1
    params['doc2vec_dm_mean'] = 1
    params['doc2vec_epochs'] = 1
    params['doc2vec_hs'] = 0
    params['doc2vec_learning_rate_start'] = 0.025
    params['doc2vec_learning_rate_end'] = 0.2
    params['doc2vec_min_count'] = 5
    params['doc2vec_negative'] = 0
    params['doc2vec_vector_size'] = 5
    params['doc2vec_window'] = 1

    # Neural network values
    params['nn_features'] = ['bpm', 'measure', 'beat']
    params['nn_batch_size'] = 100
    params['nn_dense_activation_function'] = "linear"
    params['nn_dropout'] = 0.05
    params['nn_epochs'] = 5
    params['nn_hidden_neurons'] = 10
    params['nn_layers'] = 10
    params['nn_lstm_activation_function'] = "selu"
    params['nn_lstm_n_prev'] = 4

    return params


if __name__ == '__main__':
    main()
