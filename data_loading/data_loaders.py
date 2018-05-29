import logging
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from midi_to_dataframe.midi_reader import MidiReader


class DataLoader(ABC):

    def __init__(self, encoder=None):
        self._logger = logging.getLogger(__name__)
        self._params = None
        self._encoder = encoder
        self._scaler = None

    @abstractmethod
    def load_data(self, path):
        pass

    def set_params(self, params):
        self._params = params

    def set_encoder(self, encoder):
        self._encoder = encoder

    def get_scaler(self):
        return self._scaler

    def set_scaler(self, scaler):
        self._scaler = scaler

    @staticmethod
    def frame_as_sequential(df, n_prev):
        """
        Frames a given dataframe as a sequential prediction problem in which the value of each row should be predicted
        given the previous n_prev rows.
        :param df: dataframe containing data to reform.
        :param n_prev: the number of previous rows to take into account when making predictions.
        :return: x (previous rows) and y (row to predict) mapping of values.
        """
        seq_x, seq_y = [], []
        for i in range(len(df) - n_prev):
            # Select previous n_prev rows (will be used to predict next)
            prev_rows = df.iloc[i:i + n_prev]
            seq_x.append(prev_rows.values)

            # Select row that should be predicted, given n_prev rows
            row_to_predict = df.iloc[i + n_prev]
            seq_y.append(row_to_predict.values)

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


class MidiDataLoader(DataLoader):
    # Name of the column in MIDI dataframes that should be converted to a vector representation by the encoder.
    COL_TO_VECTORIZE = "notes"
    MIDI_FILE_EXTENSION = ".mid"

    def __init__(self, note_mapper, params=None, encoder=None):
        super(MidiDataLoader, self).__init__(encoder)
        self._note_mapper = note_mapper
        self._params = params

    def load_data(self, data_source, fit_scaler=True):
        """
        Loads and prepares data from the given data source.
        :param data_source: the data source to load.
        :param fit_scaler: flag indicating whether the internal scaler object should be fit to this data. Use True for
        training data, False for test/evaluation sets (since the scaler should be previously fit on the corresponding
        training set.
        :return: the prepared dataset, separated by x and y values.
        """

        # Ensure encoder was set
        if self._encoder is None:
            self._logger.error("Unable to load data: encoder was not specified.")
            return np.vstack([]), np.vstack([])

        # Collect all MIDI files found in given data_source path
        files_to_load = self._collect_files_to_load(data_source)

        # Convert all MIDI files to Data Frames
        dataframes = []
        for midi_file in files_to_load:
            dataframes.append(self._load_midi_file(midi_file))

        # Fit scaler to values (if configured)
        if fit_scaler:
            # Fit to full concatenated set to ensure scaler sees full range of values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(pd.concat(dataframes))
            self.set_scaler(scaler)
        elif self.get_scaler() is None:
            self._logger.warning("fit_scaler disabled: scaling will be skipped.")

        # Data separated by x and y
        x_data_full = []
        y_data_full = []

        # Apply preprocessing steps to all dataframes
        for df in dataframes:

            scaler = self.get_scaler()
            if scaler is not None:
                df = pd.DataFrame(scaler.transform(df), columns=df.columns)

            (x_data, y_data) = self.frame_as_sequential(df, self._params['nn_lstm_n_prev'])

            # Append to data to return
            if len(x_data) > 0 and len(y_data) > 0:
                x_data_full.append(x_data)
                y_data_full.append(y_data)

        x_stacked = np.vstack(x_data_full)
        y_stacked = np.vstack(y_data_full)

        return x_stacked, y_stacked

    def _load_midi_file(self, path):
        """
        Loads a piece of music from a MIDI file and encodes it as a sequence of vectors and associated features, stored
        as a Dataframe.
        :param path: path to MIDI file.
        :return: Pandas Dataframe containing encoded vector sequence and configured features.
        """

        # Initialize MIDI reader with configured note mapping
        reader = MidiReader(self._note_mapper)

        # Load MIDI file as a Data Frame
        midi_dataframe = reader.convert_to_dataframe(path)

        # If empty sequence was loaded, abort
        if len(midi_dataframe) == 0:
            self._logger.warning("File '" + str(path) + "' produced empty MIDI dataframe.")
            return pd.DataFrame()

        # Drop columns not passed as features / that will not be vectorized
        cols_to_keep = self._params['nn_features'] + [self.COL_TO_VECTORIZE]
        for column in midi_dataframe:
            if column not in cols_to_keep:
                midi_dataframe.drop(column, axis=1, inplace=True)

        # Add columns to hold each element of vectors
        for index in range(0, self._params['doc2vec_vector_size']):
            midi_dataframe[str(index)] = 0

        # Convert designated column string in each row to a corresponding vector
        for index, row in midi_dataframe.iterrows():

            # Select rows that should be preserved
            values = []
            for col in cols_to_keep:
                values.append(row[col])

            # Vectorize target column
            string_to_vectorize = row[self.COL_TO_VECTORIZE]
            vector = self._encoder.convert_text_to_vector(string_to_vectorize)

            # Update row with vectorized value
            new_row = values + list(vector)
            midi_dataframe.loc[index] = new_row

        # Remove column that was vectorized
        midi_dataframe.drop(self.COL_TO_VECTORIZE, axis=1, inplace=True)

        return midi_dataframe

    def _collect_files_to_load(self, data_source):
        """
        Collects individual MIDI files to load from passed data source. Data source may be a path to an individual file,
        a directory, a list of files or a list of directories.
        :param data_source: the data source to collect files from.
        :return: list of individual files found.
        """
        files_to_load = []
        if isinstance(data_source, str):
            files_to_load += self._process_data_source(data_source)
        elif isinstance(data_source, (list,)):
            for source in data_source:
                files_to_load += self._process_data_source(source)

        return files_to_load

    def _process_data_source(self, data_source):
        """
        Processes a configured data source by adding it (for files) or all the files it contains (for directories) to
        the list of files to be loaded.
        :param data_source: the path to the data source to process.
        :return: list of files found.
        """
        files_to_load = []
        if os.path.isfile(data_source):
            files_to_load.append(data_source)
        elif os.path.isdir(data_source):
            files_to_load += list(self.find_files_in_path_by_type(data_source, self.MIDI_FILE_EXTENSION))
        return files_to_load

    def set_encoder(self, encoder):
        super(MidiDataLoader, self).set_encoder(encoder)

    def get_scaler(self):
        return super(MidiDataLoader, self).get_scaler()
