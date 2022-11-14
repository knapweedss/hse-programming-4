import os
import sys
import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
file_name = 'bert_res.pt'


def make_pool(model_res):
    return model_res[0][:, 0]


def str_to_int(string):
    """
    Конвертация в int
    """
    if string:
        return int(string)
    return 0


def best_match(query, corpus, tokenizer, model, corp, n=5):
    """
    Наилучшие 5 результатов по запросу
    """
    encoded_q = tokenizer([query], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_query_output = model(**encoded_q)
    q = make_pool(model_query_output)
    cos_sims = np.squeeze(cosine_similarity(q, corp))
    return np.array(corpus)[np.argsort(cos_sims, axis=0)[:-(n+1):-1].ravel()]


def get_result(corpus, file_name):
    print(f"Василиса Володина читает корпус...")
    if file_name == 'bert_res.pt':
        result = [max(json.loads(corpus[i])["answers"], key=lambda x: str_to_int(x["author_rating"]["value"]))["text"]
                  for i in range(50000) if json.loads(corpus[i])["answers"]]
    # создание эмбеддингов
    if not os.path.exists(f"data/{file_name}"):
        print("Лариса Гузеева не нашла эмбеддинги! Считаем...")
        encoded_answers = tokenizer(corpus[:50], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_answers)
        ans = make_pool(model_output)
        torch.save(ans, f"data/{file_name}")
        for i in tqdm(range(50, 300, 50)):
            ans = torch.load(f"data/{file_name}")
            encoded_answers_batch = tokenizer(corpus[i:i + 50],
                                          padding=True, truncation=True, max_length=24, return_tensors='pt')
            with torch.no_grad():
                model_output_batch = model(**encoded_answers_batch)
            ans = torch.cat((ans, make_pool(model_output_batch)), 0)
            torch.save(ans, f"data/{file_name}")
    ans = torch.load(f"data/{file_name}")
    if file_name == 'bert_res.pt':
        for q in search_query:
            print(f"Ларисочка Гузеева и особый гость Берт осуществляют поиск по запросу '{''.join(q)}'")
            print(*best_match(q, result, tokenizer, model, ans), sep="\n")
    return ans


if __name__ == '__main__':
    print('Получили ваш запрос..')
    parser = argparse.ArgumentParser()
    if len(sys.argv) < 5:
        raise SystemError("You should pass --path (path to file with love corpus)"
                          " and -q (what you search for) ")
    parser.add_argument("--path", help="input full path to love corpus")
    parser.add_argument('-q', '--query', action='append', nargs='+', help='Search for anything', required=True)
    args = parser.parse_args()
    data_dir = args.path  # директория с data.jsonl
    search_query = args.query  # ваш вопрос о любви
    with open(data_dir, 'r') as f:
        corpora = list(f)
    get_result(corpora, file_name)