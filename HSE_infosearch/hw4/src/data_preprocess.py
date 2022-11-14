from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation, ascii_lowercase, digits
from tqdm import tqdm
morph = MorphAnalyzer()
tokenizer = WordPunctTokenizer()
stop = set(stopwords.words("russian"))


def preprocess_text(text):
    """
    На вход подается считанный файл
    Приведение к одному регистру, удаление пунктуации и стоп-слов, лемматизация
    Возвращает предобработанные токены
    """
    clean_corpus = []
    analyze = Mystem()
    for text in tqdm(text):
        tokens = analyze.lemmatize(text)
        tokens = [t for t in tokens
                  if not any(elem in t for elem in punctuation) and
                  t not in stopwords.words("russian")]
        clean_corpus.append(" ".join(tokens).lower())
    return clean_corpus


def preprocess_text_bm25(text):
    """
    На вход подается считанный файл
    Приведение к одному регистру, удаление пунктуации и стоп-слов, лемматизация
    Возвращает предобработанные токены
    """
    tokens = tokenizer.tokenize(text.lower())
    tokens = [morph.parse(token)[0].normal_form for token in tokens
              if token not in punctuation and token not in stop
              and not set(token).intersection(digits)
              and token != " " and not set(token).intersection(ascii_lowercase)]
    return ' '.join(tokens)