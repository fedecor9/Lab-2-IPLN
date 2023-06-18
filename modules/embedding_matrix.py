import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

MAX_SEQUENCE_LENGTH = 100

class Embeddings:
    """
    A class to read the word embedding file and to create the word embedding matrix
    """
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings


    def create_embedding_matrix(self, X_train, X_eval):

        self.tokenizer.fit_on_texts(X_train)
        word_index = self.tokenizer.word_index

        x_train_sequences = self.tokenizer.texts_to_sequences(X_train)
        x_eval_sequences = self.tokenizer.texts_to_sequences(X_eval)

        x_train = pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post',
                                truncating='post')

        x_eval = pad_sequences(x_eval_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        embedding_matrix = np.zeros((len(word_index)+1, 100))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = self.word_embedding.get_vector(word)
            except:
                embedding_matrix[i] = np.zeros(100)

        return x_train, x_eval, embedding_matrix


