import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from gensim.models import KeyedVectors
import re
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Embedding, LSTM, GRU
import os

import matplotlib.pyplot as plt
import seaborn as sns
import logging

# https://keras.io/api/models/sequential/



def clean_word(word):
    """Elimina todo lo que no sea una letra, y se remueven los acentos.
    También pasa la palabra a lowercase."""
    word = word.lower()
    word = word.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    word = re.compile(r'[^a-z|ñ]').sub('', word)
    return word

set_abreviaturas = [
    ["que", r"((?<=\s)q(?=\s)|^q(?=\s)+|(?<=\s)+q$)"],
    ["por", r"((?<=\s)x(?=\s)|^x(?=\s)+|(?<=\s)+x$)"],
    ["porque", r"((?<=\s)xq(?=\s)|^xq(?=\s)+|(?<=\s)+xq$)"],
    ["porque", r"((?<=\s)pq(?=\s)|^pq(?=\s)+|(?<=\s)+pq$)"],
    ["de", r"((?<=\s)d(?=\s)|^d(?=\s)+|(?<=\s)+d$)"],
]

def change_tweet(tweet):
    tweet = re.sub(r"^U+\w+", "", tweet)
    tweet = re.sub(r"http(s)?://\S+.co(/\S+)*", "(URL)", tweet)
    tweet = re.sub(r'#\w+', 'HASHTAG', tweet)
    tweet = re.sub(r"@\w+", "USUARIO", tweet)
    tweet = re.sub(r"(!+)", "!", tweet)
    tweet = re.sub(r"([a-zA-Z]+?)\1+\b", r"\1", tweet)
    for i in range(len(set_abreviaturas)):
        tweet = re.sub(set_abreviaturas[i][1], set_abreviaturas[i][0], tweet)

    tweet = re.sub(r"jaja(ja|j|a|aj)*", "jaja", tweet, flags=re.IGNORECASE)
    return tweet

def process_data(data_set):
    train_set = []
    y_train = []
    with open(data_set, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            train_set.append(line[1])
            y_train.append(line[2])

    train_set = np.array(train_set)


    res_word_final = []
    for sentence in train_set:
        res_words = []
        for word in str(sentence).split(' '):
            # Se "limpian" y estandarizan las palabras, por ej, "Hola!" es lo mismo que "hola"
            word = clean_word(word)
            res_words.append(word)

        res_word_final.append(res_words)

    train_set = np.array([' '.join(sentence) for sentence in res_word_final])

    return train_set, y_train





class NaivesBayesTextClassifier:
    count_vect = CountVectorizer
    nb_classifier = MultinomialNB()

    def __init__(self):
        pass

    def fit(self,transform_func):
        X_train, Y_train = process_data('train.csv')

        Y_train = transform_func(np.array(Y_train))

        self.count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        x_count = self.count_vect.fit_transform(X_train)
        self.nb_classifier.fit(x_count, Y_train)


    def predict(self, X_test):
        x_test_count = self.count_vect.transform(X_test)
        return self.nb_classifier.predict(x_test_count)

# TODO
class MLPTextClassifier:
    word_embedding = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)
    mlp_classifier = MLPClassifier

    def __init__(self,hidden_layer_sizes, max_iter):

        self.mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        pass

      # Función para obtener el vector promedio de un tweet
    def get_tweet_vector(self,tweet):
        vectors = []
        for word in tweet:
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

class LSTMTextClassifier:
    word_embedding = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)
    lstm_model = None

    def __init__(self, max_sequence_length):
        self.max_sequence_length = max_sequence_length

    def get_tweet_vector(self, tweet):
        vectors = []
        for word in tweet:
            if word in self.word_embedding:
                vectors.append(self.word_embedding[word])

        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        return np.zeros(self.word_embedding.vector_size)

    def fit(self, transform_func, max_iter):
        X_train, Y_train = process_data('train.csv')

        Y_train = transform_func(np.array(Y_train))

        # Obtener las representaciones vectoriales de los tweets
        X_vect = np.array([self.get_tweet_vector(tweet) for tweet in X_train])

        X_padded = pad_sequences(X_vect, maxlen=self.max_sequence_length)

        self.lstm_model = Sequential()
        self.lstm_model.add(Embedding(input_dim=self.word_embedding.vector_size, output_dim=self.word_embedding.vector_size, trainable=False))
        self.lstm_model.add(LSTM(units=100))
        self.lstm_model.add(Dense(units=1, activation='sigmoid'))
        self.lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.lstm_model.fit(X_padded, Y_train, epochs=max_iter, batch_size=32)


    def predict(self, X_test):
        X_vect = np.array([self.get_tweet_vector(tweet) for tweet in X_test])
        X_padded = pad_sequences(X_vect, maxlen=self.max_sequence_length)
        return self.lstm_model.predict_classes(X_padded)


def main():
    #BOW estándar: se recomienda trabajar con la clase CountVectorizer de sklearn, en
    #particular, fit_transform y transform.
    # bowModel()
    wordEmbeddingsModel()
    deepLearning()

def get_max_sequence_length(tweets):
    tweet_lengths = [len(tweet.split()) for tweet in tweets]
    max_length = int(np.percentile(tweet_lengths, 90))  # Obtener el percentil 90
    return max_length

def deepLearning():
    X_train, Y_train = process_data('train.csv')

    max_sequence_length = get_max_sequence_length(X_train)  # X_train es la lista de tweets de entrenamiento

    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else -1))
    clf = LSTMTextClassifier(max_sequence_length)
    clf.fit(transform_func, 10)

    x_new, y_eval = process_data('devel.csv')
    y_eval = transform_func(np.array(y_eval))

    results = clf.predict(x_new)
    n_classes = 3
    y_true_bin = label_binarize(y_eval, classes=range(n_classes))
    y_pred_bin = label_binarize(results, classes=range(n_classes))

    print('Deep Learning')
    print("Accuracy: ", np.mean(results == y_eval))
    print("Precision: ", metrics.precision_score(y_true_bin, y_pred_bin, average='macro'))
    print("Recall: ", metrics.recall_score(y_true_bin, y_pred_bin, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_true_bin, y_pred_bin, average='macro'))

    # word2vec_model = KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt", limit=50000)
    # embedding_dim = word2vec_model.vector_size

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)

    # # Obtener el tamaño del vocabulario
    # vocab_size = len(tokenizer.word_index) + 1

    # # Convertir los textos en secuencias de números enteros
    # sequences = tokenizer.texts_to_sequences(texts)

    # # Obtener la longitud máxima de las secuencias
    # max_length = max(len(seq) for seq in sequences)

    # # Rellenar las secuencias para que todas tengan la misma longitud
    # X_train = pad_sequences(sequences, maxlen=max_length)

    # # Convertir las etiquetas a un formato adecuado para el modelo
    # y_train = np.array(labels)

    # Crear una matriz de pesos para los embeddings
    # embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # # Rellenar la matriz de pesos con los embeddings de Word2Vec
    # for word, i in tokenizer.word_index.items():
    #     if word in word2vec_model:
    #         embedding_matrix[i] = word2vec_model[word]

    # model = Sequential()

    # # Agregar la capa de embeddings
    # model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))

    # # Agregar una capa LSTM
    # model.add(LSTM(128))

    # # Agregar una capa densa para la clasificación
    # model.add(Dense(1, activation='sigmoid'))

    # # Compilar y entrenar el modelo
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=10, batch_size=32)

def wordEmbeddingsModel():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else -1))
    clf = MLPTextClassifier(hidden_layer_sizes=(50,), max_iter=500)

    clf.fit(transform_func)

    x_new, y_eval = process_data('devel.csv')
    y_eval = transform_func(np.array(y_eval))

    results = clf.predict(x_new)
    n_classes = 3
    y_true_bin = label_binarize(y_eval, classes=range(n_classes))
    y_pred_bin = label_binarize(results, classes=range(n_classes))

    print("Accuracy: ", np.mean(results == y_eval))
    print("Precision: ", metrics.precision_score(y_true_bin, y_pred_bin, average='macro'))
    print("Recall: ", metrics.recall_score(y_true_bin, y_pred_bin, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_true_bin, y_pred_bin, average='macro'))


def bowModel():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else -1))
    clf = NaivesBayesTextClassifier()

    # Process data and clean training set
    clf.fit(transform_func)

    # process devel and clean test set
    X_new, Y_eval = process_data('devel.csv')
    Y_eval = transform_func(np.array(Y_eval))

    results = clf.predict(X_new)

    n_classes = 3
    y_true_bin = label_binarize(Y_eval, classes=range(n_classes))
    y_pred_bin = label_binarize(results, classes=range(n_classes))

    print("Accuracy: ", np.mean(results == Y_eval))
    print("Precision: ", metrics.precision_score(y_true_bin, y_pred_bin, average='macro'))
    print("Recall: ", metrics.recall_score(y_true_bin, y_pred_bin, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_true_bin, y_pred_bin, average='macro'))

    # Obtener las representaciones vectoriales de los tweets

main()
