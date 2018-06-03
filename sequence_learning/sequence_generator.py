import logging


class SequenceGenerator:
    """
    Sequence Generation object.
    """

    def __init__(self, data_loader, sequence_model):
        self._logger = logging.getLogger(__name__)
        self._data_loader = data_loader
        self._sequence_model = sequence_model

    def generate(self, seed, length):
        """
        Generates a new sequence of a specified length, using a given seed sequence.
        :param seed: path to a sequence to generate the seed from.
        :param length: the length (in steps) of the sequence to generate.
        :return: generated sequence, in the form of a Pandas dataframe.
        """
        dataframes = self._data_loader.load_data(seed, fit_scaler=False, return_df=True)
        seed_df = dataframes[0]
        return self._sequence_model.generate_sequence(seed_df, self._data_loader, length)
