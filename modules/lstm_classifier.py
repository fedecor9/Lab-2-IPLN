import numpy as np
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout
from keras.layers import LSTM, Dense, Dropout, Embedding


class LSTMClassifier:

    def __init__(self, embedding_matrix, embedding_dim, max_length):

        self.lstm_model = Sequential([
            Embedding(embedding_matrix.shape[0], embedding_dim, input_length=max_length, embeddings_initializer=Constant(embedding_matrix)),
            LSTM(128),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])

    def compile_model(self):
        self.lstm_model.compile(optimizer='adam', metrics=['acc'],
        loss='sparse_categorical_crossentropy')

    def train_model(self, x_train, y_train, x_val, y_val):
        self.lstm_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, class_weight={0: 1.1, 1: 5.2})
