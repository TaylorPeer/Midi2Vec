import sys
import logging
from encoding.encoder import Encoder
from sequence_learning.sequence_learner import SequenceLearner
from data_loading.data_loaders import MidiDataLoader

from midi_to_dataframe.note_mapper import NoteMapper

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

logging.getLogger("gensim").setLevel(logging.WARNING)


def main():
    # Documents used to train semantic encoder model
    # encoder_training_docs = "../../midi-embeddings/data/full_1_measure.txt"
    encoder_training_docs = "../resources/encoder_training_docs/full_1_measure_20k.txt"

    model_params = {

        # Encoder (doc2vec) settings:
        'encoder_training_docs': encoder_training_docs,
        'doc2vec_dm': 1,
        'doc2vec_dm_mean': 1,
        'doc2vec_epochs': 2,
        'doc2vec_hs': 0,
        'doc2vec_learning_rate_start': 0.025,
        'doc2vec_learning_rate_end': 0.2,
        'doc2vec_min_count': 10,
        'doc2vec_negative': 0,
        'doc2vec_vector_size': 4,  # 24,
        'doc2vec_window': 1,  # 3,

        # Sequence learning (Keras LSTM) settings:
        'nn_features': ['bpm', 'measure', 'beat'],
        'nn_batch_size': 100,  # 25,
        'nn_dense_activation_function': "linear",
        'nn_dropout': 0,
        'nn_epochs': 5,  # 50,
        'nn_hidden_neurons': 5,  # 30,
        'nn_layers': 5,  # 15,
        'nn_lstm_activation_function': "selu",
        'nn_lstm_n_prev': 4  # 16
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
    training_docs = ["../resources/breakbeats"]

    # Load training MIDI files using MidiDataLoader
    data_loader = MidiDataLoader(note_mapper, params=model_params, encoder=encoder)
    training_data = data_loader.load_data(training_docs)

    # Train sequence learning model
    sequence_model = SequenceLearner()
    sequence_model.train(model_params, training_data)

    # TODO select seed sequence for training
    seed_sequences = ["../resources/breakbeats/084 Breakthru.mid", "../resources/breakbeats/086 Clouds.mid",
                      "../resources/breakbeats/089 Get Out.mid", "../resources/breakbeats/089 Wrong.mid",
                      "../resources/breakbeats/090 Deceive.mid", "../resources/breakbeats/090 New York.mid",
                      "../resources/breakbeats/090 Radio.mid", "../resources/breakbeats/093 Pretender.mid",
                      "../resources/breakbeats/093 Right Won.mid", "../resources/breakbeats/094 Run.mid"]

    for seed in seed_sequences:
        dataframes = data_loader.load_data(seed, fit_scaler=False, return_df=True)
        seed_df = dataframes[0]
        # Generate new sequence
        length = 32
        generated_seq_df = sequence_model.generate_sequence(seed_df, data_loader, length)
        print("---")

        print(generated_seq_df.to_string())


if __name__ == '__main__':
    main()
