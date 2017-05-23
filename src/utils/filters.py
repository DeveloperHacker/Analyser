import re

from utils.method import JavaDoc, Method
from variables.tags import *


class Filter:
    @staticmethod
    def diff(s1: str, s2: str) -> str:
        left = s1.index(s2)
        return s1[:left] + s1[left + len(s2):]

    @staticmethod
    def isKeyWord(string) -> bool:
        for keyword in TAGS:
            if keyword in string and string.index(keyword) == 0:
                diff = Filter.diff(string, keyword)
                if len(diff) == 0 or diff.isnumeric():
                    return True
        return False

    @staticmethod
    def isPunctuation(string: str) -> bool:
        return string in PUNCTUATION

    @staticmethod
    def wordsNumber(words: list) -> int:
        return len([word for word in words if Filter.isKeyWord(word) or word.isalpha()])


def filterStrings(string: str) -> str:
    return re.sub("\".*?[^\\\]\"", " %s " % string, string)


def filterTags(string: str) -> str:
    return re.sub("<[^>]*>", " %s " % HTML, string)


def filterRefs(string: str) -> str:
    return re.sub("&\w+", " %s " % REFERENCE, string)


def filterLinks(string: str) -> str:
    return re.sub("(\{@|@\{)[^\}]*\}", " %s " % LINK, string)


def filterPaths(string: str) -> str:
    return re.sub(r"(/[a-zA-Z](\w|\.|\\\s)*){2,}", " %s " % PATH, string)


def filterURLs(string: str) -> str:
    return re.sub(r"((\w+:(//|\\\\))?(\w+[\w.@]*\.[a-z]{2,3})(\w|\.|/|\\|\?|=|-)*)|(\w+:(//|\\\\))",
                  " %s " % URL, string)


def filterParamNames(string: str, params: list) -> str:
    for i, param in enumerate(params):
        string = string.replace(" %s " % param, " %s%d " % (VARIABLE, i))
    return string


def filterFunctionInvocation(string: str) -> str:
    space = r"(\s|\t)*"
    word = r"(\@|[a-zA-Z])\w*"
    param = r"({0}{1}{0}\=)?{0}{1}".format(space, word)
    regex = r"{0}{1}({0}\(({2}({0}\,{2})*)?\))+".format(space, word, param)
    return re.sub(regex, " %s " % INVOCATION, string)


def filterDotTuple(string: str) -> str:
    space = r"(\s|\t)*"
    word = r"(\@|[a-zA-Z])\w*"
    regex = r"{0}{1}(\.{1})+".format(space, word)
    return re.sub(regex, " %s " % DOT_INVOCATION, string)


def filterNumbers(string: str) -> str:
    return re.sub(r"(\+|-)?(\d+(\.|,)?\d+|\d+)", " %s " % NUMBER, string)


def filterConstants(string: str) -> str:
    string = re.sub(r"true", " %s " % TRUE, string)
    string = re.sub(r"false", " %s " % FALSE, string)
    string = re.sub(r"null", " %s " % NULL, string)
    string = re.sub(r"nil", " %s " % NULL, string)
    string = re.sub(r"none", " %s " % NULL, string)
    return string


def filterMeaninglessSentences(string: str) -> str:
    text = []
    sentence = []
    for ch in string:
        sentence.append(ch)
        if ch == '.' and len(sentence) > 1:
            sentence = "".join(sentence)
            words = [word for word in sentence.split(" ") if len(word) > 0]
            if (Filter.wordsNumber(words) / len(words)) > NORMAL_CONCENTRATION_OF_WORDS:
                text.append(sentence)
            else:
                print(sentence)
            sentence = []
    return "".join(text)


def filterStableReduction(string: str) -> str:
    return re.sub(r"([a-zA-Z])\.([a-zA-Z])\.", r" {}_\1_\2_ ".format(STABLE_REDUCTION), string)


def unpackStableReduction(string: str) -> str:
    return re.sub(r"{}_([^_]+)_([^_]+)_".format(STABLE_REDUCTION), r" \1.\2. ", string)


def expandWordsAndSymbols(string: str) -> str:
    regex = "(" + "|".join("\\" + symbol for symbol in PUNCTUATION) + ")"
    return re.sub(regex, r" \1 ", string)


def filterLongSpaces(string: str) -> str:
    return re.sub("(\s|\t)+", " ", string)


def filterFirstAndEndSpaces(string: str) -> str:
    if len(string) == 0: return string
    string = string[1:] if (string[0] == ' ') else string
    if len(string) == 0: return string
    string = string[:-1] if (string[-1] == ' ') else string
    return string


def convert(name):
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def applyFiltersForString(string: str, params: list) -> str:
    if string is None or params is None:
        raise ValueError("arguments must be not None")
    if len(string) == 0: return string
    string = string.lower()
    string = filterStrings(string)
    string = filterTags(string)
    string = filterRefs(string)
    string = filterLinks(string)
    string = filterStableReduction(string)
    string = filterURLs(string)
    string = filterNumbers(string)
    string = filterConstants(string)
    string = filterParamNames(string, params)
    string = filterFunctionInvocation(string)
    # string = filterDotTuple(string)
    string = expandWordsAndSymbols(string)
    # string = filterMeaninglessSentences(string)
    string = unpackStableReduction(string)
    string = filterLongSpaces(string)
    string = filterFirstAndEndSpaces(string)
    return string


def applyFiltersForMethod(method: Method) -> Method:
    params = method.get_param_names()
    java_doc = method.java_doc  # type: JavaDoc
    java_doc.variables = ["@variable%d %s" % (i, applyFiltersForString(convert(name).replace(r"_", " "), params)) for
                          i, name in enumerate(params)]
    java_doc.head = applyFiltersForString(java_doc.head, params)
    java_doc.params = [applyFiltersForString(param, params) for param in java_doc.params]
    java_doc.results = [applyFiltersForString(result, params) for result in java_doc.results]
    java_doc.throws = [applyFiltersForString(throw, params) for throw in java_doc.throws]
    java_doc.sees = [applyFiltersForString(see, params) for see in java_doc.sees]
    method.java_doc = java_doc
    return method
