import csv
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet') 

def get_stopwords():
    with open('stop_words_esp_anasent.csv', 'r', encoding='utf-8') as f:
        return [m.strip() for m in f.readlines()]

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

def process_data(data_set, stopwords = False, useLemas = False):
    train_set = []
    y_train = []
    with open(data_set, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            train_set.append(line[1])
            y_train.append(line[2])

    train_set = np.array(train_set)

    if stopwords:
        stopwords_set = get_stopwords()
    
    res_word_final = []
    for sentence in train_set:
        res_words = []
        for word in str(sentence).split(' '):
            # Se "limpian" y estandarizan las palabras, por ej, "Hola!" es lo mismo que "hola"
            if useLemas:
                word = WordNetLemmatizer().lemmatize(word)
            word = clean_word(word) 
            res_words.append(word)
            
        res_word_final.append(res_words)

    train_set = np.array([' '.join(sentence) for sentence in res_word_final])
    return train_set, y_train
