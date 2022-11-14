import argparse
import sys
import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.data_preprocess import preprocess_text
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

def make_corpus(texts_dir):
    """
    Создание корпуса предобработанного текста
    Возвращает список лемм
    """
    docs, l = [], []
    with open(texts_dir, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]

    for line in corpus:
        answers = json.loads(line)['answers']
        if len(answers) > 0:
            value = np.array(map(int, [ans['author_rating']['value']
                                       for ans in answers if ans != '']))
            answer = answers[np.argmax(value)]['text']
            docs.append(answer)
            l.append(preprocess_text(answer))
    return docs, l


def indexation(corpus, k=2, b=0.75):
    """
    Sparse матрица
    """

    x_count_vectorizer = count_vectorizer.fit_transform(corpus)
    x_tf_vectorizer = tf_vectorizer.fit_transform(corpus)
    tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vectorizer
    values = []
    rows = []
    cols = []
    corpus_doc_lengths = x_count_vectorizer.sum(axis=1)
    avg_doc_length = corpus_doc_lengths.mean()
    denominator_coeff = (k * (1 - b + b * corpus_doc_lengths / avg_doc_length))
    denominator_coeff = np.expand_dims(denominator_coeff, axis=-1)

    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = tf[i, j] * idf[0][j] * (k + 1)
        B = tf[i, j] + denominator_coeff[i]
        value = A / B
        values.append(value[0][0])

    return sparse.csr_matrix((values, (rows, cols)))


def query_indexation(query):
    """
    Векторизация запроса
    """
    return count_vectorizer.transform(query)


def count_bm25(query, corpus):
    """
    BM25
    """
    return corpus.dot(query.T)


def answer_love(query, corpus, names):
    """
    Функция поиска
    """
    words = preprocess_text(query)
    query_index = query_indexation(words)
    bm25 = count_bm25(query_index, corpus)
    ind = np.argsort(bm25.toarray(), axis=0)
    return np.array(names)[ind][::-1].squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if len(sys.argv) < 5:
        raise SystemError("You should pass --path (path to file with love corpus)"
                          " and -q (what you search for) ")
    parser.add_argument("--path", help="input full path to love corpus")
    parser.add_argument('-q', '--query', action='append', nargs='+', help='Search for anything', required=True)
    args = parser.parse_args()
    data_dir = args.path  # директория с data.jsonl
    search_query = args.query  # ваш вопрос о любви
    print(f"Василиса Володина обрабатывает корпус...")
    corpora, lemmas = make_corpus(data_dir)
    corpora = corpora[0:50000]
    print(f"Роза Сябитова индексирует...")
    matrix = indexation(lemmas)

    for q in search_query:
        print(f"Ларисочка Гузеева осуществляет поиск по запросу '{' '.join(q)}'")
        docs = answer_love(' '.join(q), matrix, corpora)
        print(*docs[:5], sep='\n')
