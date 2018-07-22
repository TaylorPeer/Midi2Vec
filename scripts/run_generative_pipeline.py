import sys
import logging

from data_loading import MidiDataLoader
from midi_to_dataframe import NoteMapper
from pipeline import GenerativePipeline
from evaluation import LossEvaluator, F1Evaluator

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

logging.getLogger("gensim").setLevel(logging.WARNING)


def main():
    # Documents used to train semantic encoder model
    encoder_training_docs = "../../data/1_measure_full.txt"
    # encoder_training_docs = "../resources/encoder_training_docs/full_1_measure_20k.txt"

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
        'doc2vec_vector_size': 8,
        'doc2vec_window': 5,

        # Sequence learning (Keras LSTM) settings:
        'nn_features': [],  # ['bpm', 'measure', 'beat'],
        'nn_batch_size': 16,
        'nn_dense_activation_function': "linear",
        'nn_dropout': 0,
        'nn_epochs': 1,
        'nn_hidden_neurons': 8,
        'nn_layers': 4,
        'nn_lstm_activation_function': "selu",
        'nn_lstm_n_prev': 16,
        'nn_loss': 'mean_absolute_error',
        'nn_optimizer': 'rmsprop'
    }

    # Define note mapper for MIDI file loading
    note_mapping_config_path = "../settings/map-to-group.json"
    note_mapper = NoteMapper(note_mapping_config_path)

    # Data loader used to encode MIDI-format training files
    data_loader = MidiDataLoader(note_mapper, params=pipeline_params)

    # Define training documents for sequence learning
    training_docs = ["../resources/midi/bach_chorales/01-AchGottundHerr.mid",
                     "../resources/midi/bach_chorales/02-AchLiebenChristen.mid",
                     "../resources/midi/bach_chorales/03-ChristederdubistTagundLicht.mid",
                     "../resources/midi/bach_chorales/04-ChristeDuBeistand.mid",
                     "../resources/midi/bach_chorales/05-DieNacht.mid",
                     "../resources/midi/bach_chorales/06-DieSonne.mid",
                     "../resources/midi/bach_chorales/07-HerrGott.mid",
                     "../resources/midi/bach_chorales/08-FuerDeinenThron.mid",
                     "../resources/midi/bach_chorales/09-Jesus.mid",
                     "../resources/midi/bach_chorales/10-NunBitten.mid"]

    pipeline = GenerativePipeline(params=pipeline_params)
    pipeline.set_data_loader(data_loader)
    pipeline.set_encoder_cache_dir("../notebooks/encoders")
    pipeline.set_training_docs(training_docs)
    pipeline.set_k_fold_cross_eval(k=3)
    pipeline.set_evaluator(F1Evaluator())
    # pipeline.save_best_model("models", "bach_chorales")

    result_df = pipeline.run()
    print(result_df.to_string())


if __name__ == '__main__':
    main()
