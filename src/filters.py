import re

from parts import JavaDoc


class Filter:

    NORMAL_CONCENTRATION_OF_WORDS = 0.7

    string = "@string"
    tag = "@htmlTag"
    reference = "@reference"
    link = "@link"
    url = "@url"
    variable = "@variable"
    invocation = "@invocation"
    dotInvocation = "@dotInvocation"

    keywords = {
        string,
        tag,
        reference,
        link,
        url,
        variable,
        invocation,
        dotInvocation
    }

    @staticmethod
    def isKeyWord(string) -> bool:
        return string in Filter.keywords

    @staticmethod
    def wordsNumber(words: list) -> int:
        return len([word for word in words if Filter.isKeyWord(word) or word.isalpha()])


def filterStrings(string: str) -> str:
    return re.sub("\".*?[^\\\]\"", " %s " % Filter.string, string)


def filterTags(string: str) -> str:
    return re.sub("<[^>]*>", " %s " % Filter.tag, string)


def filterRefs(string: str) -> str:
    return re.sub("&\w+", " %s " % Filter.reference, string)


def filterLinks(string: str) -> str:
    return re.sub("(\{@|@\{)[^\}]*\}", " %s " % Filter.link, string)


def filterURLs(string: str) -> str:
    return re.sub(r"([^\s]+\.[a-z]{2,3}|\w+:(//|\\\\))(\w|/|\\|\?|=|-)*", " %s " % Filter.url, string)


def filterParamNames(string: str, params: list) -> str:
    for i, param in enumerate(params):
        string = string.replace(" %s " % param, " %s%d " % (Filter.variable, i))
    return string


def filterFunctionInvocation(string: str) -> str:
    space = r"(\s|\t)*"
    word = r"(\@|[a-zA-Z])\w*"
    param = r"({0}{1}{0}\=)?{0}{1}".format(space, word)
    regex = r"{0}{1}({0}\(({2}({0}\,{2})*)?\))+".format(space, word, param)
    return re.sub(regex, " %s " % Filter.invocation, string)


def filterDotTuple(string: str) -> str:
    space = r"(\s|\t)*"
    word = r"(\@|[a-zA-Z])\w*"
    regex = r"{0}{1}(\.{1})+".format(space, word)
    return re.sub(regex, " %s " % Filter.dotInvocation, string)


def filterMeaninglessSentences(string: str) -> str:
    text = []
    sentence = []
    for ch in string:
        sentence.append(ch)
        if ch == '.' and len(sentence) > 1:
            sentence = "".join(sentence)
            words = sentence.split(" ")
            if Filter.wordsNumber(words) / len(words) > Filter.NORMAL_CONCENTRATION_OF_WORDS:
                text.append(sentence)
            sentence = []
    return "".join(text)


def filterLongSpaces(string: str) -> str:
    return re.sub("(\s|\t)+", " ", string)


def filterFirstAndEndSpaces(string: str) -> str:
    if len(string) == 0: return string
    string = string[1:] if (string[0] == ' ') else string
    if len(string) == 0: return string
    string = string[:-1] if (string[-1] == ' ') else string
    return string


def applyFiltersForString(string: str, params: list) -> str:
    string = filterStrings(string)
    string = filterTags(string)
    string = filterRefs(string)
    string = filterLinks(string)
    string = filterURLs(string)
    string = filterParamNames(string, params)
    string = filterFunctionInvocation(string)
    # string = filterDotTuple(string)
    string = filterMeaninglessSentences(string)
    string = filterLongSpaces(string)
    string = filterFirstAndEndSpaces(string)
    return string


def applyFiltersForMethods(methods: list) -> list:
    for method in methods:
        params = [param.name for param in method.params]
        javaDoc = method.javaDoc  # type: JavaDoc
        javaDoc.head = applyFiltersForString(javaDoc.head, params)
        javaDoc.params = [applyFiltersForString(param, params) for param in javaDoc.params]
        javaDoc.results = [applyFiltersForString(result, params) for result in javaDoc.results]
        javaDoc.throws = [applyFiltersForString(throw, params) for throw in javaDoc.throws]
        javaDoc.sees = [applyFiltersForString(see, params) for see in javaDoc.sees]
        method.javaDoc = javaDoc
    return methods
