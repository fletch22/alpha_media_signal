import pickle


def save(obj: object, file_path: str):
    with open(file_path, 'wb') as out_file:
        pickle.dump(obj, out_file)


def load(file_path):
    with open(file_path, 'rb') as in_file:
        obj = pickle.load(in_file)

    return obj
