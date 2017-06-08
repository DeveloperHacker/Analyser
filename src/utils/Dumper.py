import _pickle as pickle
import json


def pkl_dump(instance, path: str):
    with open(path, "wb") as file:
        pickle.dump(instance, file)


def pkl_load(path: str):
    with open(path, "rb") as file:
        instance = pickle.load(file)
    return instance


def json_dump(instance, path: str):
    with open(path, "w") as file:
        json.dump(instance, file)


def json_load(path: str):
    with open(path, "r") as file:
        instance = json.load(file)
    return instance
