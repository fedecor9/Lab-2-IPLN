import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from modules.clean_words import process_data


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
