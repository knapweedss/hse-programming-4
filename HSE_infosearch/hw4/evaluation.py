import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from bert_search import str_to_int
from src.data_preprocess import preprocess_text
from bert_search import get_result
from bm25_search import indexation, query_indexation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False)
tfidf_vectorizer = TfidfVectorizer(use_idf=True)

data_dir = './lovecorpus/data.jsonl' # директория с data.jsonl
if not os.path.exists("data/bert_answers.pt") or not os.path.exists("data/bert_questions.pt"):
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


def compute_score(evaluation_matrix, n=5):
    matrix_sorted = np.argsort(evaluation_matrix, axis=1)[:, :-(n+1):-1]
    q_range = np.expand_dims(np.arange(evaluation_matrix.shape[0]), axis=1)
    q_res = np.sum(q_range == matrix_sorted, axis=1)
    return np.sum(q_res)/evaluation_matrix.shape[0]


def bm25_eval(query, corpus):
    corpus_matrix = indexation(corpus)
    query_matrix = query_indexation(query)
    return np.dot(query_matrix, corpus_matrix.T).toarray()


if __name__ == '__main__':
    with open(data_dir, 'r') as f:
        corpus_str = "[" + ",".join(f.readlines()) + "]"
        corpus = json.loads(corpus_str)

    answer_texts = [max(corpus[i]["answers"], key=lambda x: str_to_int(x["author_rating"]["value"]))["text"]
                    for i in range(10000) if corpus[i]["answers"]]

    question_texts = [question["question"] + " " + question["comment"]
                      for question in corpus[:10000] if question["answers"]]

    # bm25
    print('Предобработка данных для BM25...')
    bm25_matrix = bm25_eval(preprocess_text(question_texts)[:10000],
                            preprocess_text(answer_texts)[:10000])
    bm25_score_top5 = compute_score(bm25_matrix)
    bm25_score_top = compute_score(bm25_matrix, 1)
    # bert
    file_name = 'bert_answers.pt'
    if os.path.exists(f"data/{file_name}"):
        bert_answer_emb = torch.load(f"data/{file_name}")
    else:
        bert_answer_emb = get_result(answer_texts, file_name)
    file_name2 = 'bert_questions.pt'
    if os.path.exists(f"data/{file_name2}"):
        bert_q_emb = torch.load(f"data/{file_name2}")
    else:
        bert_q_emb = get_result(question_texts, file_name2)
    bert_question_embs_batch = bert_q_emb[:10000]
    bert_answer_embs_batch = bert_answer_emb[:10000]
    bert_cos_sim = cosine_similarity(bert_question_embs_batch, bert_answer_embs_batch)
    bert_score_top = compute_score(bert_cos_sim, 1)
    bert_score_top5 = compute_score(bert_cos_sim)

    print(f"BERT: n=5: {bert_score_top5}, n=1: {bert_score_top}")
    print(f"BM25: n=5: {bm25_score_top5}, n=1: {bm25_score_top}")
