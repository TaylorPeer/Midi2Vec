import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from midi_to_dataframe import MidiReader

from data_loading.data_loader import DataLoader


class MidiDataLoader(DataLoader):
    """
    Data loader for MIDI music files.
    """

    # Name of the column in MIDI dataframes that should be converted to a vector representation
    COL_TO_VECTORIZE = "notes"

    MIDI_FILE_EXTENSION = ".mid"

    def __init__(self, note_mapper, params=None, encoder=None):
        super(MidiDataLoader, self).__init__(encoder)
        self._note_mapper = note_mapper
        self._params = params

        # Initialize MIDI reader with configured note mapping
        self._reader = MidiReader(self._note_mapper)

    def load_data_as_df(self, data_source, fit_scaler=True):
        """
        Loads and prepares data from the given data source, returning arrays of x and y values.
        :param data_source: the data source to load.
        :param fit_scaler: flag indicating whether the internal scaler object should be fit to this data. Use True for
                training data, False for test/evaluation sets (since the scaler should be previously fit on the corresponding
                training set.
        :return: arrays of x and y values, prepared as configured.
        """

        if self._encoder is None:
            self._logger.error("Unable to load data: encoder was not specified.")
            return pd.DataFrame()

        dataframes = self._load_raw_data(data_source)
        self._apply_scaler(dataframes, fit_scaler)

        return dataframes

    def load_data_as_array(self, data_source, fit_scaler=True, pred_labels=False):
        """
        Loads and prepares data from the given data source, returning arrays of x and y values.
        :param data_source: the data source to load.
        :param fit_scaler: flag indicating whether the internal scaler object should be fit to this data. Use True for
                training data, False for test/evaluation sets (since the scaler should be previously fit on the corresponding
                training set.
        :param pred_labels: TODO
        :return: arrays of x and y values, prepared as configured.
        """

        dataframes = self.load_data_as_df(data_source, fit_scaler)
        if len(dataframes) == 0:
            return np.vstack([]), np.vstack([])

        # Data separated by x and y
        x_data_full = []
        y_data_full = []

        # Retrieve classification labels from folder structure (if configured)
        labels = []
        if pred_labels:
            files_to_load = self._collect_files_to_load(data_source)

            # Extract label from name of (deepest) directory containing file
            for file in files_to_load:
                path = os.path.dirname(os.path.normpath(file))
                label = os.path.basename(path)
                labels.append(label)

        # TODO explain whats happening in this loop
        for index, df in enumerate(dataframes):

            pred_label = None
            if pred_labels and len(labels) > 0:
                pred_label = labels[index]

            # Transform data into sequences of equal length for training
            (x_data, y_data) = self.frame_as_sequential(df,
                                                        n_prev=self._params['nn_lstm_n_prev'],
                                                        pred_label=pred_label)

            # Append to data to return
            if len(x_data) > 0 and len(y_data) > 0:
                x_data_full.append(x_data)
                y_data_full.append(y_data)

        x_stacked = np.vstack(x_data_full)
        y_stacked = np.vstack(y_data_full)

        return x_stacked, y_stacked

    def _load_raw_data(self, data_source):
        """
        Collects and loads raw data sources as dataframes.
        :param data_source: the data to load.
        :return: collection of dataframes loaded from the source data.
        """
        files_to_load = self._collect_files_to_load(data_source)
        return self._load_files_as_dataframes(files_to_load)

    def _apply_scaler(self, dataframes, fit_scaler):
        """
        Applies scaling to a dataset. Optionally fits the scaler to this set or re-uses a previously fit scaler.
        :param dataframes: the collection of dataframes to be scaled
        :param fit_scaler: flag indicating if a new scaler object should be created and fit to this set.
        :return: None.
        """
        if fit_scaler:
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit to concatenated set to ensure scaler sees full range of values
            scaler.fit(pd.concat(dataframes))
            self.set_scaler(scaler)
        elif self.get_scaler() is not None:
            scaler = self.get_scaler()
        else:
            self._logger.warning("fit_scaler disabled: scaling will be skipped.")
            return

        # TODO check that all dataframes have len > 0

        # Apply scaler
        for index, df in enumerate(dataframes):
            dataframes[index] = pd.DataFrame(scaler.transform(df), columns=df.columns)
        return

    def _load_midi_file(self, path):
        """
        Loads a piece of music from a MIDI file and encodes it as a sequence of vectors and associated features, stored
        as a Dataframe.
        :param path: path to MIDI file.
        :return: Pandas Dataframe containing encoded vector sequence and configured features.
        """

        self._configure_fields_to_extract()

        # Load MIDI file as a Data Frame
        midi_dataframe = self._reader.convert_to_dataframe(path)

        # If empty sequence was loaded, abort
        if len(midi_dataframe) == 0:
            self._logger.warning("File '" + str(path) + "' produced empty MIDI dataframe.")
            return pd.DataFrame()

        # Add columns to hold each element of vectors
        for index in range(0, self._params['doc2vec_vector_size']):
            midi_dataframe[str(index)] = 0

        # Convert designated column string in each row to an equivalent vector representation
        cols_to_keep = self._params['nn_features']
        new_rows = []
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
            new_rows.append(new_row)

        new_df = pd.DataFrame(new_rows)
        return new_df

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

    def _load_files_as_dataframes(self, files_to_load):
        # TODO cache is only valid for same nn_features! -> use nn_features as part of ID
        dataframes = []
        for midi_file in files_to_load:
            encoder_id = self.get_encoder().get_id()
            if (encoder_id, midi_file) in self._df_cache:
                df = self._df_cache[(encoder_id, midi_file)]
                self._logger.debug("Loaded MIDI DF from cache for file: " + str(midi_file))
            else:
                df = self._load_midi_file(midi_file)
                self._df_cache[(encoder_id, midi_file)] = df
            dataframes.append(df)
        return dataframes

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

    def _configure_fields_to_extract(self):
        """
        Configures the fields to extract when loading MIDI files.
        :return: None
        """
        # Configure values MidiReader should extract
        # TODO constants
        if 'timestamp' not in self._params['nn_features']:
            self._reader.set_extract_timestamp(False)
        if 'bpm' not in self._params['nn_features']:
            self._reader.set_extract_bpm(False)
        if 'time_signature' not in self._params['nn_features']:
            self._reader.set_extract_time_signature(False)
        if 'measure' not in self._params['nn_features']:
            self._reader.set_extract_measure(False)
        if 'beat' not in self._params['nn_features']:
            self._reader.set_extract_beat(False)

    def set_encoder(self, encoder):
        super(MidiDataLoader, self).set_encoder(encoder)

    def get_scaler(self):
        return super(MidiDataLoader, self).get_scaler()
