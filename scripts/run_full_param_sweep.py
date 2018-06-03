import sys
import logging

from data_loading import MidiDataLoader
from midi_to_dataframe import NoteMapper
from pipeline import Pipeline
from optimization import BruteForce

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def main():
    # Documents used to train semantic encoder model
    encoder_training_docs = "../resources/encoder_training_docs/full_1_measure_20k.txt"

    param_sweep_values = {

        # Encoder (doc2vec) settings:
        'doc2vec_docs': [encoder_training_docs],
        'doc2vec_dm': [1],
        'doc2vec_dm_mean': [1],
        'doc2vec_epochs': [1],
        'doc2vec_hs': [0],
        'doc2vec_learning_rate_start': [0.025],
        'doc2vec_learning_rate_end': [0.2],
        'doc2vec_min_count': [5],
        'doc2vec_negative': [0],
        'doc2vec_vector_size': [20],
        'doc2vec_window': [1],

        # Sequence learning (Keras LSTM) settings:
        'nn_features': [['bpm', 'measure', 'beat']],
        'nn_batch_size': [100],
        'nn_dense_activation_function': ["linear"],
        'nn_dropout': [0],
        'nn_epochs': [10],
        'nn_hidden_neurons': [10],
        'nn_layers': [10],
        'nn_lstm_activation_function': ["selu"],
        'nn_lstm_n_prev': [4]
    }

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Data loader used to encode MIDI-format training files
    data_loader = MidiDataLoader(note_mapper)

    # Define training documents for sequence learning
    training_docs = ["../resources/midi/breakbeats/084 Breakthru.mid",
                     "../resources/midi/breakbeats/086 Clouds.mid",
                     "../resources/midi/breakbeats/089 Get Out.mid",
                     "../resources/midi/breakbeats/089 Wrong.mid",
                     "../resources/midi/breakbeats/090 Deceive.mid",
                     "../resources/midi/breakbeats/090 New York.mid",
                     "../resources/midi/breakbeats/090 Radio.mid",
                     "../resources/midi/breakbeats/093 Pretender.mid",
                     "../resources/midi/breakbeats/093 Right Won.mid",
                     "../resources/midi/breakbeats/094 Run.mid"]

    pipeline = Pipeline()
    pipeline.set_data_loader(data_loader)
    pipeline.set_training_docs(training_docs)
    pipeline.set_k_fold_cross_eval(k=5)

    brute_force_param_sweep = BruteForce(params=param_sweep_values)
    brute_force_param_sweep.save_best_model("../notebooks/models", "test")
    pipeline.set_optimizer(brute_force_param_sweep)

    results_df = pipeline.run()
    print(results_df.to_string())


if __name__ == '__main__':
    main()
