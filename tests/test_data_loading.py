import sys
import logging
import unittest
import pandas as pd
import numpy as np

from midi_to_dataframe import NoteMapper

from encoding import Encoder
from data_loading import MidiDataLoader

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logging.getLogger("gensim").setLevel(logging.WARNING)


class MidiDataLoadingTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MidiDataLoadingTests, self).__init__(*args, **kwargs)
        self.test_params = {"doc2vec_dm": 1,
                            "doc2vec_dm_mean": 1,
                            "doc2vec_epochs": 1,
                            "doc2vec_hs": 0,
                            "doc2vec_learning_rate_start": 0.025,
                            "doc2vec_learning_rate_end": 0.01,
                            "doc2vec_min_count": 2,
                            "doc2vec_negative": 0,
                            "doc2vec_vector_size": 4,
                            "doc2vec_window": 1,
                            "nn_features": ['bpm', 'measure', 'beat'],
                            "nn_lstm_n_prev": 16}

        note_mapping_config_path = "resources/data_loading/map-to-group.json"
        note_mapper = NoteMapper(note_mapping_config_path)

        encoder = self._get_test_encoder()
        self.test_instance = MidiDataLoader(note_mapper, params=self.test_params, encoder=encoder)

    def test_data_preparation(self):
        data = np.array([['', '1', '2', '3'],
                         ['a', '1a', '2a', '3a'],
                         ['b', '1b', '2b', '3b'],
                         ['c', '1c', '2c', '3c'],
                         ['d', '1d', '2d', '3d']])

        df = pd.DataFrame(data=data[1:, 1:],
                          index=data[1:, 0],
                          columns=data[0, 1:])

        (x, y) = self.test_instance.frame_as_sequential(df, 1)
        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 3)

        (x, y) = self.test_instance.frame_as_sequential(df, 2)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)

        (x, y) = self.test_instance.frame_as_sequential(df, 3)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)

        (x, y) = self.test_instance.frame_as_sequential(df)
        self.assertEqual(len(x), 1)
        self.assertEqual(len(y), 1)

        return

    def test_load_midi_file(self):
        df = self.test_instance.load_data("resources/data_loading/freestyler-clip.MID", return_df=True)
        print(df[0].head())
        # TODO assert something useful

    def _get_test_encoder(self):
        """
        Trains and returns a test encoder module, prepared from test documents.
        :return: trained encoder module for testing.
        """
        encoder = Encoder(self.test_params)
        docs = encoder.load_documents("resources/encoding/test_docs.line")
        encoder.set_documents(docs)
        encoder.train()
        return encoder


def main():
    unittest.main()


if __name__ == '__main__':
    main()
