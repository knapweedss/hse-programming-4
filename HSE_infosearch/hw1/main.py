import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.load_dataset import get_filepath
from src.data_preprocess import preprocess_text
curr_dir = os.getcwd()
data_dir = curr_dir + '/friends-data'
vectorizer = CountVectorizer(analyzer='word')


def read_files(fpath):
    """
    чтение файлов
    """
    with open(fpath, 'r') as f:
        text = f.read()
    return text


def make_corpus(texts_dirs):
    """
    Создание корпуса предобработанного текста
    Возвращает список лемм
    """
    corpus = []
    for text in texts_dirs:
        corpus.append(preprocess_text(read_files(text)))
    return corpus


def indexation(vectorizer, corpus):
    """
    Индексация
    """
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
    return df


def answer_questions(data):
    """
    Осуществляет поиск и отвечает на вопросы
    """
    cast = {}
    matrix_freq = np.asarray(data.to_numpy().sum(axis=0)).ravel()
    names = data.columns.to_numpy()
    sorted_matrix = sorted(list(range(len(matrix_freq))), key=lambda num: matrix_freq[num])[::-1]
    print("Самое частое слово:", vectorizer.get_feature_names_out()[sorted_matrix[0]])
    print("Самое редкое слово:", vectorizer.get_feature_names_out()[sorted_matrix[-1]])
    cast["моника"] = np.sum(matrix_freq[(names == "моника") | (names == "мон")])
    cast["рэйчел"] = np.sum(matrix_freq[(names == "рэйчел") | (names == "рейч")])
    cast["чендлер"] = np.sum(matrix_freq[(names == "чендлер") | (names == "чэндлер") | (names == "чен")])
    cast["фиби"] = np.sum(matrix_freq[(names == "фиби") | (names == "фибс")])
    cast["росс"] = matrix_freq[names == "росс"][0]
    cast["джоуи"] = np.sum(matrix_freq[(names == "джоуи") | (names == "джои") | (names == "джо")])
    common_words = data.loc[:, (data != 0).all(axis=0) == True].columns.to_list()
    print("Набор слов, который есть во всех документах коллекции:", ", ".join(common_words))
    print('Самый популярный гг:', max(cast, key=cast.get).title())


if __name__ == '__main__':
    corpora = make_corpus(get_filepath(data_dir))
    answer_questions(indexation(vectorizer, corpora))
