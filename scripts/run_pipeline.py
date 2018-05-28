import sys
import logging

from data_loading.midi_data_loader import MidiDataLoader
from midi_to_dataframe.note_mapper import NoteMapper
from pipeline.pipeline import Pipeline

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def main():
    # Documents used to train semantic encoder model
    encoder_training_docs = "/Users/taylorpeer/Projects/se-project/midi-embeddings/data/full_1_measure_100k.txt"

    pipeline_params = {

        # Encoder (doc2vec) settings:
        'doc2vec_docs': encoder_training_docs,
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

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Data loader used to encode MIDI-format training files
    data_loader = MidiDataLoader(pipeline_params, note_mapper)

    # Define training documents for sequence learning
    training_docs = []
    training_docs.append("resources/breakbeats/084 Breakthru.mid")
    training_docs.append("resources/breakbeats/086 Clouds.mid")
    training_docs.append("resources/breakbeats/089 Get Out.mid")
    training_docs.append("resources/breakbeats/089 Wrong.mid")
    training_docs.append("resources/breakbeats/090 Deceive.mid")
    training_docs.append("resources/breakbeats/090 New York.mid")
    training_docs.append("resources/breakbeats/090 Radio.mid")
    training_docs.append("resources/breakbeats/093 Pretender.mid")
    training_docs.append("resources/breakbeats/093 Right Won.mid")
    training_docs.append("resources/breakbeats/094 Run.mid")

    # Define evaluation documents for sequence learning
    evaluation_docs = []
    evaluation_docs.append("/Users/taylorpeer/Projects/se-project/midi-embeddings/data/corpora/test/test")

    pipeline = Pipeline(params=pipeline_params)

    pipeline.set_data_loader(data_loader)

    pipeline.set_training_docs(training_docs)
    # pipeline.set_test_docs(evaluation_docs)
    pipeline.set_k_fold_cross_eval(k=5)

    # pipeline.set_optimizer(genetic)

    metrics = pipeline.run()

    print(metrics.get_precision())
    print(metrics.get_recall())
    print(metrics.get_f1())


if __name__ == '__main__':
    main()
