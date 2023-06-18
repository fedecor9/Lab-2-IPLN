import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences



class Embeddings:
    """
    A class to read the word embedding file and to create the word embedding matrix
    """
    tokenizer = Tokenizer()

    def __init__(self, word_embeddings, max_length):
        self.word_embeddings = word_embeddings
        self.max_length = max_length


    def create_embedding_matrix(self, X_train, X_eval):

        self.tokenizer.fit_on_texts(X_train)
        word_index = self.tokenizer.word_index
        vocab_size = len(word_index) + 1

        x_train_sequences = self.tokenizer.texts_to_sequences(X_train)
        x_eval_sequences = self.tokenizer.texts_to_sequences(X_eval)

        x_train = pad_sequences(x_train_sequences, maxlen= self.max_length, padding='post',
                                truncating='post')

        x_eval = pad_sequences(x_eval_sequences, maxlen= self.max_length, padding='post', truncating='post')

        embedding_matrix = np.zeros(shape=(vocab_size, 300))
        print('Shape of embedding matrix: ', embedding_matrix.shape)
        print('Shape word embeddings: ', len(self.word_embeddings.keys()))
        for word, i in word_index.items():
            if word in self.word_embeddings:
                # get first 47 dimensions of the word embedding
                embedding_matrix[i] = self.word_embeddings[word]

        return x_train, x_eval, embedding_matrix


