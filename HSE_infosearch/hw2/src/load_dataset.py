import os


def get_filepath(data_dir):
    """
    Полученние абсолютного пути до файла
    На вход подается текущая рабочая директория
    Возвращает список всех .txt файлов папки friends-data
    и названия текстов (для выввода после поиска)
    """
    friends_sub, docs = [], []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                friends_sub.append(os.path.join(root, file))
                docs.append(file[0:-7])
    return friends_sub, docs

