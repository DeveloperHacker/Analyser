import _pickle as pickle


def dump(instance, path: str):
    with open(path, "wb") as file:
        pickle.dump(instance, file)


def load(path: str):
    with open(path, "rb") as file:
        instance = pickle.load(file)
    return instance
