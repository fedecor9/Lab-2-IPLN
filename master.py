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


def main():
    #BOW estándar: se recomienda trabajar con la clase CountVectorizer de sklearn, en
    #particular, fit_transform y transform.
    # bowModel()
    # wordEmbeddingsModel()
    deep_learning()


def deep_learning():
    transform_func = np.vectorize(lambda x: 1 if x == 'P' else 0 )

    # Conjunto de entrenamiento pasado a word embeddings
    X_train, Y_train = process_data('train.csv')
    Y_train = transform_func(np.array(Y_train))

    # Conjunto de validación
    X_eval, Y_eval = process_data('devel.csv')
    Y_eval = transform_func(np.array(Y_eval))

    wv = WordEmbeddings()
    word_embeddings = wv.load_embeddings(X_train)

    # Crear embedding matrix
    em = Embeddings(word_embeddings)
    X_train, X_eval, embedding_matrix = em.create_embedding_matrix(X_train, X_eval)

    lstm_model = LSTMClassifier(embedding_matrix, 100)

    lstm_model.compile_model()

    lstm_model.train_model(X_train, Y_train, X_eval, Y_eval)


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
