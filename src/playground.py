from contants import JAVA_DOC, CONTRACT, DESCRIPTION
from utils import dumpers

HASH = "hash"


def hash_repr(method):
    description = method[DESCRIPTION]
    name = description["name"]
    owner = description["owner"]
    result = type_repr(description["type"])
    parameters = description["parameters"]
    parameters = [(parameter["name"], type_repr(parameter["type"])) for parameter in parameters]
    parameters = ", ".join("%s: %s" % parameter for parameter in parameters)
    description = "%s %s : %s(%s) %s" % (DESCRIPTION, owner, name, parameters, result)
    return description


def type_repr(string):
    if string == 'B': return "byte"
    if string == 'C': return "char"
    if string == 'D': return "double"
    if string == 'F': return "float"
    if string == 'I': return "int"
    if string == 'J': return "long"
    if string == 'S': return "short"
    if string == 'Z': return "boolean"
    if string == 'V': return "void"
    if string == "void*": return "void"
    return string.split("<")[0]


def convert():
    methods_v1 = dumpers.json_load("resources/data-sets/joda-time-v1.json")
    methods_v2 = dumpers.json_load("resources/data-sets/joda-time-v2.json")
    for method in methods_v1:
        method[HASH] = hash_repr(method)
    for method in methods_v2:
        method[HASH] = hash_repr(method)
    methods = []
    for method_v1 in methods_v1:
        for i, method_v2 in enumerate(methods_v2):
            if method_v1[HASH] == method_v2[HASH]:
                method = {
                    JAVA_DOC: method_v2[JAVA_DOC],
                    DESCRIPTION: method_v2[DESCRIPTION],
                    CONTRACT: method_v1[CONTRACT]}
                methods.append(method)
                del methods_v2[i]
                break
        else:
            dumpers.json_print(method_v1)
    dumpers.json_dump(methods, "resources/data-sets/joda-time-out.json")


def shuffle():
    import random
    methods = dumpers.json_load("resources/data-sets/joda-time.json")
    random.shuffle(methods)
    dumpers.json_dump(methods, "resources/data-sets/joda-time-s.json")


if __name__ == '__main__':
    shuffle()
