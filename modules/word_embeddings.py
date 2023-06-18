
import numpy as np
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from modules.clean_words import process_data


class WordEmbeddings():
    word_embedding = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)

    def __init__(self):
        pass

    # Función para obtener el vector promedio de un tweet
    def get_tweet_vector(self, tweet):
        vectors = {}
        tweet = tweet.split()
        for word in tweet:
            if self.word_embedding.__contains__(word):
                vectors[word] = self.word_embedding.__getitem__(word)

        return vectors

    def load_embeddings(self, x_train):
        # Obtener las representaciones vectoriales de los tweets
        # return np.array([self.get_tweet_vector(tweet) for tweet in x_train])
        return self.word_embedding

