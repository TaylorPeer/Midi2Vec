import sys
import logging
from encoding import Encoder
from sequence_learning import GenerativeSequenceLearner, SequenceGenerator
from data_loading import MidiDataLoader

from midi_to_dataframe import MidiWriter, NoteMapper

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

logging.getLogger("gensim").setLevel(logging.WARNING)


def main():
    # Documents used to train semantic encoder model
    #encoder_training_docs = "../../midi-embeddings/data/full_1_measure.txt"
    encoder_training_docs = "../resources/encoder_training_docs/full_1_measure_20k.txt"

    model_params = {

        # Encoder (doc2vec) settings:
        'doc2vec_docs': encoder_training_docs,
        'doc2vec_dm': 1,
        'doc2vec_dm_mean': 1,
        'doc2vec_epochs': 2,
        'doc2vec_hs': 0,
        'doc2vec_learning_rate_start': 0.025,
        'doc2vec_learning_rate_end': 0.2,
        'doc2vec_min_count': 10,
        'doc2vec_negative': 0,
        'doc2vec_vector_size': 20,  # 24,
        'doc2vec_window': 10,  # 3,

        # Sequence learning (Keras LSTM) settings:
        'nn_features': ['bpm', 'measure', 'beat'],
        'nn_batch_size': 15,
        'nn_dense_activation_function': "linear",
        'nn_dropout': 0.1,
        'nn_epochs': 75,
        'nn_hidden_neurons': 30,  # 30,
        'nn_layers': 20,  # 15,
        'nn_lstm_activation_function': "selu",
        'nn_lstm_n_prev': 16,
        'nn_loss': 'mean_absolute_error',
        'nn_optimizer': 'rmsprop'
    }

    # Train encoder
    encoder = Encoder(model_params)
    docs = encoder.load_documents(model_params['doc2vec_docs'])
    encoder.set_documents(docs)
    encoder.train()

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Define training documents for sequence learning
    training_docs = ["../resources/midi/breakbeats"]

    # Load training MIDI files using MidiDataLoader
    data_loader = MidiDataLoader(note_mapper, params=model_params, encoder=encoder)
    training_data = data_loader.load_data_as_array(training_docs)

    # Train sequence learning model
    sequence_model = GenerativeSequenceLearner(model_params)
    sequence_model.train(training_data)

    # TODO select seed sequence for training
    seed_sequences = ["../resources/midi/breakbeats/084 Breakthru.mid",
                      "../resources/midi/breakbeats/086 Clouds.mid",
                      "../resources/midi/breakbeats/089 Get Out.mid",
                      "../resources/midi/breakbeats/089 Wrong.mid",
                      "../resources/midi/breakbeats/090 Deceive.mid",
                      "../resources/midi/breakbeats/090 New York.mid",
                      "../resources/midi/breakbeats/090 Radio.mid",
                      "../resources/midi/breakbeats/093 Pretender.mid",
                      "../resources/midi/breakbeats/093 Right Won.mid",
                      "../resources/midi/breakbeats/094 Run.mid"]

    sequence_generator = SequenceGenerator(data_loader, sequence_model)
    length = 64

    for seq_index, seed in enumerate(seed_sequences):
        generated_seq_df = sequence_generator.generate(seed, length)

        writer = MidiWriter(note_mapper)
        save_to_path = "test_seq_" + str(seq_index) + ".mid"
        writer.convert_to_midi(generated_seq_df, save_to_path)
        print("---")

        print(generated_seq_df.to_string())


if __name__ == '__main__':
    main()
