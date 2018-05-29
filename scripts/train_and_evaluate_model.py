import sys
import logging
from encoding.encoder import Encoder
from evaluation.evaluator import Evaluator
from sequence_learning.sequence_learner import SequenceLearner
from data_loading.data_loaders import MidiDataLoader

from midi_to_dataframe.note_mapper import NoteMapper

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def main():
    # Documents used to train semantic encoder model
    encoder_training_docs = "resources/encoder_training_docs/full_1_measure_20k.txt"

    model_params = {

        # Encoder (doc2vec) settings:
        'encoder_training_docs': encoder_training_docs,
        'doc2vec_dm': 1,
        'doc2vec_dm_mean': 1,
        'doc2vec_epochs': 1,
        'doc2vec_hs': 0,
        'doc2vec_learning_rate_start': 0.025,
        'doc2vec_learning_rate_end': 0.2,
        'doc2vec_min_count': 5,
        'doc2vec_negative': 0,
        'doc2vec_vector_size': 5,
        'doc2vec_window': 1,

        # Sequence learning (Keras LSTM) settings:
        'nn_features': ['bpm', 'measure', 'beat'],
        'nn_batch_size': 100,
        'nn_dense_activation_function': "linear",
        'nn_dropout': 0.05,
        'nn_epochs': 5,
        'nn_hidden_neurons': 10,
        'nn_layers': 10,
        'nn_lstm_activation_function': "selu",
        'nn_lstm_n_prev': 4
    }

    # Train encoder
    encoder = Encoder(model_params)
    docs = encoder.load_documents(model_params['encoder_training_docs'])
    encoder.set_documents(docs)
    encoder.train()

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Define training documents for sequence learning
    training_docs = ["/Users/taylorpeer/Projects/se-project/midi-embeddings/data/corpora/test/training"]

    # Define evaluation documents for sequence learning
    evaluation_docs = []
    evaluation_docs.append("/Users/taylorpeer/Projects/se-project/midi-embeddings/data/corpora/test/test")

    # Load training MIDI files using MidiDataLoader
    data_loader = MidiDataLoader(note_mapper, params=model_params, encoder=encoder)
    training_data = data_loader.load_data(training_docs)

    # Set fit_scaler=False to re-use scaler from training set
    test_data = data_loader.load_data(evaluation_docs, fit_scaler=False)
    (x_test, y_test) = test_data

    # Train sequence learning model
    model = SequenceLearner.train_model(model_params, training_data)

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

    # Remove doc2vec_docs params setting, since otherwise params can't be printed
    model_params = dict((key, value) for key, value in model_params.items() if key != 'doc2vec_docs')

    print(str(model_params))
    print("- precision: " + str(precision))
    print("- recall: " + str(recall))
    print("- f1: " + str(f1))
    print("- average error: " + str(average_error))
    print("---")


if __name__ == '__main__':
    main()
