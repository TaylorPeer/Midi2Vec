import sys
import logging
import unittest
import random
from encoding import Encoder

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class EncoderTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(EncoderTests, self).__init__(*args, **kwargs)

    def test_train_doc2vec_model(self):
        test_model_params = {"doc2vec_dm": 1,
                             "doc2vec_dm_mean": 1,
                             "doc2vec_epochs": 1,
                             "doc2vec_hs": 0,
                             "doc2vec_learning_rate_start": 0.025,
                             "doc2vec_learning_rate_end": 0.01,
                             "doc2vec_min_count": 2,
                             "doc2vec_negative": 0,
                             "doc2vec_vector_size": 1,
                             "doc2vec_window": 1}

        # Train encoder
        encoder = Encoder(test_model_params)
        docs = encoder.load_documents("resources/encoding/test_docs.line")
        encoder.set_documents(docs)
        encoder.train()

        # Check model returns a random word containing 2 underscores (instrument_note_duration)
        random_word = random.choice(encoder.get_word_vectors().index2word)
        self.assertTrue(random_word.count("_") == 2)

        # TODO test convert_vector_to_text

        return


def main():
    unittest.main()


if __name__ == '__main__':
    main()
