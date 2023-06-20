
import numpy as np
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from modules.clean_words import process_data


class MLPTextClassifier:
    word_embedding = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)
    mlp_classifier = MLPClassifier

    def __init__(self,hidden_layer_sizes, max_iter):
        self.mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, solver='sgd', learning_rate='adaptive')
        pass

    # Función para obtener el vector promedio de un tweet
    def get_tweet_vector(self,tweet):
        vectors = []
        for word in tweet.split(' '):
            if word in self.word_embedding:
                vectors.append(self.word_embedding[word])

        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        
        return np.zeros(self.word_embedding.vector_size)

    def fit(self,transform_func):
        X_train, Y_train = process_data('train.csv')

        Y_train = transform_func(np.array(Y_train))

        # Obtener las representaciones vectoriales de los tweets
        X_vect = np.array([self.get_tweet_vector(tweet) for tweet in X_train])

        self.mlp_classifier.fit(X_vect, Y_train)

    def predict(self, X_test):
        x_test_count = np.array([self.get_tweet_vector(tweet) for tweet in X_test])
        return self.mlp_classifier.predict(x_test_count)

