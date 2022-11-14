import os


def get_filepath(data_dir):
    """
    Полученние абсолютного пути до файла
    На вход подается текущая рабочая директория
    Возвращает список всех .txt файлов папки friends-data
    """
    friends_sub = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                friends_sub.append(os.path.join(root, file))
    return friends_sub


