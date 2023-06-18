
import numpy as np
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from modules.clean_words import process_data


class WordEmbeddings():
    word_embedding = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)

    def __init__(self):
        pass

    def load_embeddings(self, x_train):
        # Obtener las representaciones vectoriales de los tweets
        # return self.word_embedding

        vocab = [word for tweet in x_train for word in tweet.split()]
        word_vec_dict= {}
        for word in vocab:
            if self.word_embedding.has_index_for(word):
                word_vec_dict[word]= self.word_embedding.get_vector(word)
        return word_vec_dict

