import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from modules.clean_words import process_data
from keras.utils import pad_sequences
import scipy.sparse as sp

def get_stopwords():
    with open('stop_words_esp_anasent.csv', 'r', encoding='utf-8') as f:
        return [m.strip() for m in f.readlines()]

class NaivesBayesTextClassifier:
    count_vect = CountVectorizer
    nb_classifier = MultinomialNB()

    def __init__(self):
        pass

    def fit(self, transform_func, usePositiveWords=False, useStopWords=False):
        X_train, Y_train = process_data('train.csv', useLemas=True)
        Y_train = transform_func(np.array(Y_train))
        if (useStopWords):
            self.count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=get_stopwords())
        else:
            self.count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        x_count = self.count_vect.fit_transform(X_train)
        if (usePositiveWords):         
            x_count = self.add_positive_features(X_train, x_count)
            x_count = self.add_negative_features(X_train, x_count)
        self.nb_classifier.fit(x_count, Y_train)

    def add_positive_features(self, X_train, x_count):
        positive_words = []
        with open('lexico_pos_lemas_grande.csv', newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                positive_words.append(line[0])
        print(len(positive_words))
        positive_attributes = []
        for  tweet in X_train:
            positive_words_tweet = 0
            for word_in_tweet in tweet.split(' '):
                if (word_in_tweet in positive_words):
                    positive_words_tweet += 1
            positive_attributes.append(positive_words_tweet)

        positive_attributes = np.array(positive_attributes).reshape(-1, 1)
        positive_attributes_matrix = sp.csr_matrix(positive_attributes)
            # Create  np array  with concatenation of x_count and positive_attributes
        x_count = np.hstack((x_count.toarray(), positive_attributes_matrix.toarray()))
        return x_count
    
    def add_negative_features(self, X_train, x_count):
        positive_words = []
        with open('lexico_neg_lemas_grande.csv', newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for line in reader:
                positive_words.append(line[0])
        print(len(positive_words))
        positive_attributes = []
        for  tweet in X_train:
            positive_words_tweet = 0
            for word_in_tweet in tweet.split(' '):
                if (word_in_tweet in positive_words):
                    positive_words_tweet += 1
            positive_attributes.append(positive_words_tweet)

        positive_attributes = np.array(positive_attributes).reshape(-1, 1)
        positive_attributes_matrix = sp.csr_matrix(positive_attributes)
            # Create  np array  with concatenation of x_count and positive_attributes
        x_count = np.hstack((x_count, positive_attributes_matrix.toarray()))
        return x_count


    def predict(self, X_test, usePositiveWords=False):
       
        x_test_count = self.count_vect.transform(X_test)
        if (usePositiveWords):
            x_test_count = self.add_positive_features(X_test, x_test_count)
            x_test_count = self.add_negative_features(X_test, x_test_count)
        return self.nb_classifier.predict(x_test_count)
