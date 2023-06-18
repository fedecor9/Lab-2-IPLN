import numpy as np
from gensim.models import KeyedVectors
from modules.clean_words import process_data
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding


class LSTMClassifier:

    def __init__(self, embedding_matrix, embedding_dim):
        self.lstm_model = Sequential([
            Embedding(10000, embedding_dim, input_length=100,
            weights=[embedding_matrix]),
            LSTM(128),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])

    def compile_model(self):
        self.lstm_model.compile(optimizer='adam', metrics=['acc'],
        loss='sparse_categorical_crossentropy')

    def train_model(self, x_train, y_train, x_val, y_val):
        self.lstm_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
