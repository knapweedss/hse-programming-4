import argparse
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.load_dataset import get_filepath
from src.data_preprocess import preprocess_text


parser = argparse.ArgumentParser()
if len(sys.argv) < 5:
    raise SystemError("You should pass --path (path to file with Friends subs)"
                      " and -q (word to search for) ")
parser.add_argument("--path", help="input path to Friends Data folder")
parser.add_argument('-q', '--query', action='append', help='<Required> Set flag', required=True)
args = parser.parse_args()
data_dir = args.path  # директория с субтитрами Друзей
search_query = args.query  # поисковой запрос
vectorizer = TfidfVectorizer()


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


def indexation(corpus):
    """
    Индексация
    """
    X = vectorizer.fit_transform([' '.join(i) for i in corpus])
    return X.toarray()


def query_indexation(query):
    """
    Векторизация запроса
    """
    return vectorizer.transform([' '.join(query)]).toarray()


def cos_similarity(query, corpus):
    """
    Считает косинусную близость
    """
    return cosine_similarity(query, corpus)[0]


def friends_search(query, corpus, doc_names):
    """
    Функция поиска
    """
    words = preprocess_text(query)
    query_index = query_indexation(words)
    final_index = np.argsort(cos_similarity(query_index, corpus))
    return np.array(doc_names)[final_index][::-1]


if __name__ == '__main__':
    corpora, documents = get_filepath(data_dir)
    corpora = make_corpus(corpora)
    matrix = indexation(corpora)
    for q in search_query:
        print(f"Осуществляется поиск по запросу '{q}'")
        docs = friends_search(q, matrix, documents)
        print(*docs, sep='\n')
    print('Поиск завершен')



