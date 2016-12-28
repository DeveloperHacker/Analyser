
def generateTextSet(methods: list, absoluteFileName: str):
    sentences = set()
    for method in methods:
        head = method.javaDoc.head  # type: str
        params = method.javaDoc.params  # type: list
        results = method.javaDoc.results  # type: list
        throws = method.javaDoc.throws  # type: list
        sees = method.javaDoc.sees  # type: list
        sentences.add(head)
        sentences.update(set(params))
        sentences.update(set(results))
        sentences.update(set(throws))
        sentences.update(set(sees))
    for sentence in sentences:
        if len(sentence) == 0:
            sentences.remove(sentence)
            break
    file = open(absoluteFileName, "w")
    file.write("\n".join(sentences))
    return sentences
