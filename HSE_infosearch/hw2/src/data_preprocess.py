from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation, ascii_lowercase, digits
mystem = Mystem()
stopwords = stopwords.words("russian")
new_stopwords = ["это"]
stopwords.extend(new_stopwords)


def preprocess_text(text):
    """
    На вход подается считанный файл
    Приведение к одному регистру, удаление пунктуации и стоп-слов, лемматизация
    Возвращает предобработанные токены
    """
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in stopwords
              and not set(token).intersection(digits)
              and not set(token).intersection(ascii_lowercase)
              and token != " " and token not in punctuation]
    return tokens
