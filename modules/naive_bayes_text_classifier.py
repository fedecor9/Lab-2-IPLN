import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from modules.clean_words import process_data
from keras.utils import pad_sequences


class NaivesBayesTextClassifier:
    count_vect = CountVectorizer
    nb_classifier = MultinomialNB()

    def __init__(self):
        pass

    def fit(self,transform_func, usePositiveWords=False):
        X_train, Y_train = process_data('train.csv',useLemas=True)

        Y_train = transform_func(np.array(Y_train))

        self.count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        x_count = self.count_vect.fit_transform(X_train)

        if (usePositiveWords):         
            positive_words = []
            positive_attributes = []
            with open('lexico_pos_lemas_grande.csv', newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    positive_words.append(line[0])
            print(len(positive_words))
            
            for word in positive_words:
                if word in self.count_vect.vocabulary_:
                    column_index = self.count_vect.vocabulary_[word]
                    attribute_values = x_count[:, column_index].toarray().flatten()
                    positive_attributes.append(attribute_values)
                else:
                    # Si la palabra positiva no está presente en el vocabulario BoW,
                    # asignar atributos de ceros para esa palabra
                    positive_attributes.append([0] * x_count.shape[0])

            positive_attributes_matrix = np.zeros((x_count.shape[0], len(positive_attributes)))
            # Asignar los valores de positive_attributes a la matriz de ceros
            for i, attribute_values in enumerate(positive_attributes):
                positive_attributes_matrix[:, i] = attribute_values
                
            x_count = np.concatenate((x_count.toarray(), positive_attributes_matrix), axis=1)
            
        
        self.nb_classifier.fit(x_count, Y_train)
        
        # negative_words = []
        # with open('lexico_neg_lemas_grande.csv', newline='', encoding="utf-8") as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for line in reader:
        #         negative_words.append(line[0])
        # print(len(negative_words))


    def predict(self, X_test, usePositiveWords=False):
       
        x_test_count = self.count_vect.transform(X_test)
        if (usePositiveWords):
            positive_words = []
            positive_attributes = []
            with open('lexico_pos_lemas_grande.csv', newline='', encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for line in reader:
                    positive_words.append(line[0])
            print(len(positive_words))
            
            for word in positive_words:
                if word in self.count_vect.vocabulary_:
                    column_index = self.count_vect.vocabulary_[word]
                    attribute_values = x_test_count[:, column_index].toarray().flatten()
                    positive_attributes.append(attribute_values)
                else:
                    # Si la palabra positiva no está presente en el vocabulario BoW,
                    # asignar atributos de ceros para esa palabra
                    positive_attributes.append([0] * x_test_count.shape[0])

            positive_attributes_matrix = np.zeros((x_test_count.shape[0], len(positive_attributes)))
            # Asignar los valores de positive_attributes a la matriz de ceros
            for i, attribute_values in enumerate(positive_attributes):
                positive_attributes_matrix[:, i] = attribute_values
                
            x_test_count = np.concatenate((x_test_count.toarray(), positive_attributes_matrix), axis=1)
        return self.nb_classifier.predict(x_test_count)
