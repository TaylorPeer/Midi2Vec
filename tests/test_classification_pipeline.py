import numpy as np
import sys
import logging
import unittest
from pipeline import ClassificationPipeline

logger = logging.getLogger()
logger.level = logging.INFO
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ClassificationPipelineTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ClassificationPipelineTests, self).__init__(*args, **kwargs)
        self._test_instance = ClassificationPipeline()

    def test(self):
        self._test_instance._fit_label_encoder(np.array(['country', 'jazz', 'disco', 'techno']))
        encoded = self._test_instance._encode_labels(np.array(['country', 'jazz', 'jazz', 'disco', 'techno']))
        print(encoded)
        # decoded = self._test_instance._decode_predictions(encoded)
        # print(decoded)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
