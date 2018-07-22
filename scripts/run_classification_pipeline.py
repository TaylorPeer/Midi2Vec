import sys
import logging

from data_loading import MidiDataLoader
from midi_to_dataframe import NoteMapper
from pipeline import ClassificationPipeline
from evaluation import LossEvaluator

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

logging.getLogger("gensim").setLevel(logging.WARNING)


def main():
    # Documents used to train semantic encoder model
    # encoder_training_docs = "../../midi-embeddings/data/1_measure_full.txt"
    encoder_training_docs = "../resources/encoder_training_docs/full_1_measure_20k.txt"

    pipeline_params = {

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
        'doc2vec_vector_size': 8,  # 24,
        'doc2vec_window': 10,  # 3,

        # Sequence learning (Keras LSTM) settings:
        'nn_features': ['bpm', 'measure', 'beat'],
        'nn_batch_size': 128,
        'nn_dense_activation_function': "linear",
        'nn_dropout': 0,
        'nn_epochs': 1,
        'nn_hidden_neurons': 8,  # 30,
        'nn_layers': 4,  # 15,
        'nn_lstm_activation_function': "selu",
        'nn_lstm_n_prev': 1024,
        'nn_loss': 'mean_absolute_error',
        'nn_optimizer': 'rmsprop'
    }

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Data loader used to encode MIDI-format training files
    data_loader = MidiDataLoader(note_mapper, params=pipeline_params)

    # Define training documents for sequence learning
    training_docs = ["/Users/taylorpeer/Projects/se-project/Midi2Vec/resources/classification/fold1",
                     "/Users/taylorpeer/Projects/se-project/Midi2Vec/resources/classification/fold2",
                     "/Users/taylorpeer/Projects/se-project/Midi2Vec/resources/classification/fold3"]

    pipeline = ClassificationPipeline(params=pipeline_params)
    pipeline.set_data_loader(data_loader)
    pipeline.set_training_docs(training_docs)
    pipeline.set_k_fold_cross_eval(k=3)
    pipeline.set_evaluator(LossEvaluator())
    pipeline.run()


if __name__ == '__main__':
    main()
