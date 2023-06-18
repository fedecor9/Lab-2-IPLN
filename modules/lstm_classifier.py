import numpy as np
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D


class LSTMClassifier:

    def __init__(self, embedding_matrix, embedding_dim, max_length):
        self.lstm_model = Sequential([
            Embedding(embedding_matrix.shape[0], embedding_dim, input_length=max_length, embeddings_initializer=Constant(embedding_matrix)),
            SpatialDropout1D(0.2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(3, activation='softmax'),
        ])

    def compile_model(self):
        self.lstm_model.compile(optimizer='adam', metrics=['acc'], loss='categorical_crossentropy')

    def train_model(self, x_train, y_train, x_val, y_val):
        self.lstm_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5,batch_size=128)


