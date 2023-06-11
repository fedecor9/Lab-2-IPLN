import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Word2Vec
import pandas as pd
import re 

# model = Word2Vec.load("Spanish Billion Word Corpus")

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

# # Función para obtener el vector de una palabra
# def get_word_vector(word):
#     if word in model.wv.vocab:
#         return model.wv[word]
#     else:
#         # Si la palabra no está en el vocabulario, devuelve un vector de ceros
#         return np.zeros(model.vector_size)

# # Función para obtener el vector promedio de un tweet
# def get_tweet_vector(tweet):
#     # Tokenizar el tweet en palabras
#     words = tweet.split()
#     # Obtener los vectores de las palabras y almacenarlos en una lista
#     word_vectors = [get_word_vector(word) for word in words]
#     # Calcular el vector promedio utilizando la función mean de numpy
#     tweet_vector = np.mean(word_vectors, axis=0)
#     return tweet_vector
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
        


def main():
    #BOW estándar: se recomienda trabajar con la clase CountVectorizer de sklearn, en
    #particular, fit_transform y transform.
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else -1))
    clf = NaivesBayesTextClassifier()
    
    # Process data and clean training set
    clf.fit(transform_func)

    # process devel and clean test set
    X_new, Y_eval = process_data('test.csv')
    Y_eval = transform_func(np.array(Y_eval))
    
    results = clf.predict(X_new)
    print("Accuracy: ", np.mean(results == Y_eval))
    # for i, (tweet, prediction) in enumerate(zip(X_new, results)):
    #     print(f"Tweet: {tweet} | Prediction: {prediction} | Real: {Y_eval[i]}")
    #     if i == 10:
    #         break


main()
