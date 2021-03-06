import logging
import os
import numpy as np
from abc import ABC


class DataLoader(ABC):

    def __init__(self, encoder=None):
        self._logger = logging.getLogger(__name__)
        self._params = None
        self._encoder = encoder
        self._scaler = None
        self._df_cache = {}

    def set_params(self, params):
        self._params = params

    def get_encoder(self):
        return self._encoder

    def set_encoder(self, encoder):
        self._encoder = encoder

    def get_scaler(self):
        return self._scaler

    def set_scaler(self, scaler):
        self._scaler = scaler

    @staticmethod
    def frame_as_sequential(df, n_prev=None, pred_label=None):
        """
        Frames a dataframe as a sequential prediction problem in which the value of each row should be predicted
        given the previous n_prev rows.
        :param df: dataframe containing data to reform.
        :param n_prev: the number of previous rows to take into account when making predictions.
        :param pred_label: classification label (for classification only)
        :return: x (previous rows) and y (row to predict) mapping of values.
        """

        if n_prev is None:
            n_prev = len(df) - 1

        seq_x, seq_y = [], []
        for i in range(len(df) - n_prev):
            # Select previous n_prev rows (will be used to predict next)
            prev_rows = df.iloc[i:i + n_prev]
            seq_x.append(prev_rows.values)

            if pred_label is not None:
                row_to_predict = np.array([pred_label])
            else:
                # Select row that should be predicted, given n_prev rows
                row_to_predict = df.iloc[i + n_prev].values

            seq_y.append(row_to_predict)

        x_vals = np.array(seq_x)
        y_vals = np.array(seq_y)

        return x_vals, y_vals

    @staticmethod
    def find_files_in_path_by_type(path, file_type):
        """
        Finds all MIDI files in a directory of a given type/extension.
        :param path: the directory to retrieve files from.
        :param file_type: the file extension to filter by.
        :return: the full paths to all matching files found.
        """
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(file_type):
                    yield os.path.join(root, file)
