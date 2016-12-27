
def generateVocabulary(methods: list, absoluteFileName: str):
    words = set()
    print(len(methods))
    for method in methods:
        head = method.javaDoc.head  # type: str
        params = method.javaDoc.params  # type: list
        results = method.javaDoc.results  # type: list
        throws = method.javaDoc.throws  # type: list
        sees = method.javaDoc.sees  # type: list
        words.update({word for word in head.split(" ") if len(word) > 0})
        words.update({word for param in params for word in param.split(" ") if len(word) > 0})
        words.update({word for result in results for word in result.split(" ") if len(word) > 0})
        words.update({word for throw in throws for word in throw.split(" ") if len(word) > 0})
        words.update({word for see in sees for word in see.split(" ") if len(word) > 0})
    file = open(absoluteFileName, "w")
    for word in words: file.write(word + '\n')
