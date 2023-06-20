import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize

# import modules
from modules.clean_words import process_data
from modules.naive_bayes_text_classifier import NaivesBayesTextClassifier
from modules.mpl_text_classifier import MLPTextClassifier
from modules.lstm_classifier import LSTMClassifier
from modules.embedding_matrix import Embeddings
from modules.word_embeddings import WordEmbeddings
from pysentimiento import create_analyzer

def get_stopwords():
    with open('stop_words_esp_anasent.csv', 'r', encoding='utf-8') as f:
        return set( m.strip() for m in f.readlines())

def main():
    #BOW estándar: se recomienda trabajar con la clase CountVectorizer de sklearn, en
    #particular, fit_transform y transform.
    # bowModel()
    wordEmbeddingsModel()
    # pysentimiento_model()
    # deep_learning()

def pysentimiento_model():
    analyzer = create_analyzer(task="sentiment", lang="es")   
    X_test, Y_test = process_data('test.csv')
    predictions = analyzer.predict(X_test)
    for tweet, prediction in zip(X_test, predictions):
        print(f"Tweet: {tweet}")
        print(f"Clasificación: {max(prediction['labels'])}")
        print('\n')

def deep_learning():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else 2))
    # Conjunto de entrenamiento pasado a word embeddings
    X_train, Y_train = process_data('train.csv')
    Y_train = transform_func(np.array(Y_train))

    # Tweet más largo
    max_tweet_length = max([len(tweet.split()) for tweet in X_train])
    
    # Conjunto de validación
    X_eval, Y_eval = process_data('devel.csv')
    Y_eval = transform_func(np.array(Y_eval))

    n_classes = 3
    Y_train = label_binarize(Y_train, classes=range(n_classes))
    Y_eval = label_binarize(Y_eval, classes=range(n_classes))

    wv = WordEmbeddings()
    word_embeddings = wv.load_embeddings(X_train)

    # Crear embedding matrix
    em = Embeddings(word_embeddings, max_tweet_length)
    X_train, X_eval, embedding_matrix = em.create_embedding_matrix(X_train, X_eval)

    lstm_model = LSTMClassifier(embedding_matrix, 300, max_tweet_length)

    lstm_model.compile_model()

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_eval = np.array(X_eval)
    Y_eval = np.array(Y_eval)

    lstm_model.train_model(X_train, Y_train, X_eval, Y_eval)

    

def wordEmbeddingsModel():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else 2))
    clf = MLPTextClassifier(hidden_layer_sizes=(100,), max_iter=2500)

    clf.fit(transform_func)

    x_new, y_eval = process_data('devel.csv')
    y_eval = transform_func(np.array(y_eval))

    results = clf.predict(x_new)
    n_classes = 3
    y_true_bin = label_binarize(y_eval, classes=range(n_classes))
    y_pred_bin = label_binarize(results, classes=range(n_classes))

    print("Macro Accuracy: ", np.mean(results == y_eval))
    print("Macro Precision: ", metrics.precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0))
    print("Macro Recall: ", metrics.recall_score(y_true_bin, y_pred_bin, average='macro',zero_division=0))
    
    f1 = metrics.f1_score(y_true_bin, y_pred_bin, average='macro')
    print("Macro F1-score:", f1)

    f1 = metrics.f1_score(y_true_bin, y_pred_bin, average=None)
    class_labels = ["NEG", "POS", "NONE"]

    for i, f1_value in enumerate(f1):
      print(f"F1-score para clase {class_labels[i]}: {f1_value}")


def bowModel():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else (0 if x == 'N' else 2))
    clf = NaivesBayesTextClassifier()

    # Process data and clean training set
    clf.fit(transform_func, usePositiveWords=True, useStopWords=True)

    # process devel and clean test set
    X_new, Y_eval = process_data('devel.csv', useLemas=True)
    Y_eval = transform_func(np.array(Y_eval))

    
    # Create 2d np array from X_train

    results = clf.predict(X_new, usePositiveWords=True)

    n_classes = 3
    y_true_bin = label_binarize(Y_eval, classes=range(n_classes))
    y_pred_bin = label_binarize(results, classes=range(n_classes))

    print("Accuracy: ", np.mean(results == Y_eval))
    print("Precision: ", metrics.precision_score(y_true_bin, y_pred_bin, average='macro'))
    print("Recall: ", metrics.recall_score(y_true_bin, y_pred_bin, average='macro'))
    print("F1 Score: ", metrics.f1_score(y_true_bin, y_pred_bin, average='macro'))

    # Obtener las representaciones vectoriales de los tweets


main()
