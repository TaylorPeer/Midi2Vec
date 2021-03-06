import re
import logging
import time
import multiprocessing
import gensim
import gensim.models.doc2vec
from pathlib import Path
from gensim.models import Doc2Vec

from scipy import spatial
from collections import namedtuple

TrainingDocument = namedtuple('TrainingDocument', 'words tags')


class Encoder:
    """
    Class for learning vector representations of string values and converting between strings and those vectors.
    """

    # TODO make this configurable
    TOKEN_DELIMITER = ","

    MAX_COSINE_DISTANCE = 2

    def __init__(self, params):
        self._logger = logging.getLogger(__name__)
        self._id = self.generate_id(params)
        self._params = params
        self._docs = []
        self._model = None
        self._document_vectors = {}

    def set_model(self, model):
        self._model = model

    def save(self, save_to_path):
        """
        Saves the (trained) Doc2Vec model.
        :param save_to_path: the path where the model should be stored.
        :return: None.
        """
        if self._model is None:
            self._logger.error("Encoder could not be saved because it is null.")
        else:
            self._model.save(save_to_path)
            self._logger.info("Encoder saved to: " + str(save_to_path))

    def load(self, path_to_stored_model):
        """
        Loads a (trained) Doc2Vec model from disk.
        :param path_to_stored_model: the path to load from.
        :return: None.
        """
        self._model = Doc2Vec.load(path_to_stored_model)
        self._logger.info("Encoder loaded from: " + str(path_to_stored_model))

    @staticmethod
    def load_if_exists(path, encoder_id):
        path_to_check = path + "/" + encoder_id
        encoder_path = Path(path_to_check)
        if encoder_path.is_file():
            return Doc2Vec.load(path_to_check)
        return None

    def convert_text_to_vector(self, text):
        """
        Convert string into a corresponding vector representation.
        :param text: the string to convert to a vector.
        :return: the vector representation.
        """
        # Check vector cache if text has already been vectorized before
        if text not in self._document_vectors:
            tokens = text.split(self.TOKEN_DELIMITER)
            vector = self._model.infer_vector(tokens)
            self._document_vectors[text] = vector
        else:
            vector = self._document_vectors[text]

        return vector

    def convert_feature_vectors_to_text(self, vectors):
        """
        Converts feature vectors into their corresponding textual representations.
        :param vectors: the vectors to convert to text.
        :return: a list of text representations corresponding to the input vectors.
        """
        # Skip first n features, which are not part of document vector
        index = len(self._params['nn_features'])

        values = []
        for vector in vectors:
            value = self._get_most_similar_vector(vector[index:])
            values.append(value)

        return values

    def convert_feature_vector_to_text(self, vector):
        """
        Converts feature vector into a corresponding textual representation.
        :param vector: the vector to convert to text.
        :return: text representation corresponding to the input vector.
        """
        # Skip first n features, which are not part of document vector
        index = len(self._params['nn_features'])

        return self._get_most_similar_vector(vector[index:])

    def train(self):
        """
        Trains the document vector model as configured.
        :return: None.
        """
        assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

        # TODO ensure that params and docs were set

        # Model parameters:
        cores = multiprocessing.cpu_count()
        dm = self._params['doc2vec_dm']
        dm_mean = self._params['doc2vec_dm_mean']
        end_alpha = self._params['doc2vec_learning_rate_end']
        epochs = self._params['doc2vec_epochs']
        hs = self._params['doc2vec_hs']
        negative = self._params['doc2vec_negative']
        min_count = self._params['doc2vec_min_count']
        start_alpha = self._params['doc2vec_learning_rate_start']
        vector_size = self._params['doc2vec_vector_size']
        window = self._params['doc2vec_window']

        self._logger.info("Training encoder model...")

        start = time.time()

        # Create model
        self._model = Doc2Vec(dm=dm,
                              dm_mean=dm_mean,
                              vector_size=vector_size,
                              window=window,
                              negative=negative,
                              hs=hs,
                              min_count=min_count,
                              workers=cores)

        self._model.build_vocab(self._docs)

        # Train model
        self._model.train(self._docs,
                          total_examples=len(self._docs),
                          epochs=epochs,
                          start_alpha=start_alpha,
                          end_alpha=end_alpha)

        end = time.time()
        message = "Trained encoder model in " + str(end - start) + " seconds"
        self._logger.info(message)

    def set_documents(self, docs):
        """
        Sets the documents to use for model training.
        :param docs: the documents.
        :return: None.
        """
        self._docs = docs

    def load_documents(self, path_to_documents):
        """
        Loads documents from a newline separated text file.
        :param path_to_documents: the path to the file to load from.
        :return: the loaded documents.
        """
        docs = []
        processed_doc_count = 0
        with open(path_to_documents, 'rb') as data:
            for line_no, line in enumerate(data, 1):
                tokens = [x.strip() for x in gensim.utils.to_unicode(line).split(Encoder.TOKEN_DELIMITER)]
                words = tokens[0:]
                tags = [line_no]
                docs.append(TrainingDocument(words, tags))
                processed_doc_count += 1
                if processed_doc_count % 100000 == 0:
                    self._logger.info("Loaded " + str(processed_doc_count) + " documents")

        return docs

    def _get_most_similar_vector(self, query):
        """
        Finds the most similar vector for a query among the document vector cache.
        :param query: the query vector.
        :return: the most similar vector in the cache.
        """
        most_similar = ""
        most_similar_distance = self.MAX_COSINE_DISTANCE
        for values, vector in self._document_vectors.items():
            distance = spatial.distance.cosine(query, vector)
            if distance < most_similar_distance:
                most_similar = values
                most_similar_distance = distance
        return most_similar

    @staticmethod
    def generate_id(params):
        """
        Generates an ID that encapsulates the set of encoder parameters given.
        :param params: a dictionary containing the model parameters.
        :return: the encoder ID.
        """
        # Get the doc2vec model parameters
        # (exclude the doc2vec_docs tuple, since this is handled separately )
        encoder_params = dict(
            (key, value) for key, value in params.items() if key.startswith("doc2vec_") and key != 'doc2vec_docs')

        # TODO doc2vec_docs string should be part of ID

        # Create unique ID from parameters
        encoder_param_vals = [(Encoder._shorten_param_name(key) + "_" + str(value)) for key, value in
                              encoder_params.items()]
        encoder_id = '-'.join(sorted(encoder_param_vals))

        return encoder_id

    @staticmethod
    def _shorten_param_name(name):
        name = re.sub("doc2vec_", "", name)
        name = re.sub("_", "", name)
        return name

    def get_id(self):
        return self._id

    def get_word_vectors(self):
        return self._model.wv
