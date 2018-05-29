import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec

from scipy import spatial
from collections import namedtuple

import logging
import time
import multiprocessing

TrainingDocument = namedtuple('TrainingDocument', 'words tags')


class Encoder:
    """
    Class for learning vector representations of string values and converting between strings and those vectors.
    """

    TOKEN_DELIMITER = ","

    def __init__(self, params):
        self._logger = logging.getLogger(__name__)
        self._id = self.generate_id(params)
        self._params = params
        self._docs = []
        self._model = None
        self._document_vectors = {}

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

    def convert_vectors_to_text(self, vectors):
        """
        Convert vectors into their corresponding textual representations.
        :param vectors: the vectors to convert to text.
        :return: a list of text representations corresponding to the input vectors.
        """
        # Skip first n features, which are not part of doc2vec vector
        # TODO handle this outside of this method
        index = len(self._params['nn_features'])

        values = []
        for vector in vectors:
            # Lookup most similar vector encountered in training set
            values.append(self._get_most_similar_vector(vector[index:]))

        return values

    def train(self):
        """
        Trains the document vector model as configured.
        :return: None.
        """
        assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

        # TODO check if params and docs were set

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

        start = time.clock()

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

        end = time.clock()
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
        most_similar_distance = 999  # TODO
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

        # Create unique ID from parameters
        encoder_id = frozenset(encoder_params.items())

        return encoder_id

    def get_id(self):
        return self._id

    def get_word_vectors(self):
        return self._model.wv
