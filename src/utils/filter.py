import re

from utils.method import JavaDoc, Method


class Filter:
    NORMAL_CONCENTRATION_OF_WORDS = 0.7

    string = "@string"
    tag = "@html"
    reference = "@reference"
    link = "@link"
    path = "@url"
    url = "@url"
    variable = "@variable"
    invocation = "@invocation"
    dotInvocation = "@dotInvocation"
    number = "@number"
    stableReduction = "@sr"
    head = "@head"
    param = "@param"
    result = "@return"
    see = "@see"

    keywords = {
        string,
        tag,
        reference,
        link,
        path,
        url,
        variable,
        invocation,
        dotInvocation,
        number,
        stableReduction,
        head,
        param,
        result,
        see
    }

    punctuation = {
        ".",
        ",",
        ":",
        ";",
        "(",
        ")",
        "{",
        "}"
    }

    @staticmethod
    def diff(s1: str, s2: str) -> str:
        left = s1.index(s2)
        return s1[:left] + s1[left + len(s2):]

    @staticmethod
    def isKeyWord(string) -> bool:
        for keyword in Filter.keywords:
            if keyword in string and string.index(keyword) == 0:
                diff = Filter.diff(string, keyword)
                if len(diff) == 0 or diff.isnumeric():
                    return True
        return False

    @staticmethod
    def isPunctuation(string: str) -> bool:
        return string in Filter.punctuation

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


def filterPaths(string: str) -> str:
    return re.sub(r"(/[a-zA-Z](\w|\.|\\\s)*){2,}", " %s " % Filter.path, string)


def filterURLs(string: str) -> str:
    return re.sub(r"((\w+:(//|\\\\))?(\w+[\w.@]*\.[a-z]{2,3})(\w|\.|/|\\|\?|=|-)*)|(\w+:(//|\\\\))", " %s " % Filter.url, string)


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


def filterNumbers(string: str) -> str:
    return re.sub(r"(\+|-)?(\d+(\.|,)?\d+|\d+)", " %s " % Filter.number, string)


def filterMeaninglessSentences(string: str) -> str:
    text = []
    sentence = []
    for ch in string:
        sentence.append(ch)
        if ch == '.' and len(sentence) > 1:
            sentence = "".join(sentence)
            words = [word for word in sentence.split(" ") if len(word) > 0]
            if (Filter.wordsNumber(words) / len(words)) > Filter.NORMAL_CONCENTRATION_OF_WORDS:
                text.append(sentence)
            else:
                print(sentence)
            sentence = []
    return "".join(text)


def filterStableReduction(string: str) -> str:
    return re.sub(r"([a-zA-Z])\.([a-zA-Z])\.", r" {}_\1_\2_ ".format(Filter.stableReduction), string)


def unpackStableReduction(string: str) -> str:
    return re.sub(r"{}_([^_]+)_([^_]+)_".format(Filter.stableReduction), r" \1.\2. ", string)


def expandWordsAndSymbols(string: str) -> str:
    return re.sub(r"(\{})".format("|\\".join(Filter.punctuation)), r" \1 ", string)


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
    if len(string) == 0: return string
    string = filterStrings(string)
    string = filterTags(string)
    string = filterRefs(string)
    string = filterLinks(string)
    string = filterStableReduction(string)
    string = filterURLs(string)
    string = filterNumbers(string)
    string = filterParamNames(string, params)
    string = filterFunctionInvocation(string)
    # string = filterDotTuple(string)
    string = expandWordsAndSymbols(string)
    # string = filterMeaninglessSentences(string)
    string = unpackStableReduction(string)
    string = string.lower()
    string = filterLongSpaces(string)
    string = filterFirstAndEndSpaces(string)
    return string


def applyFiltersForMethod(method: Method) -> Method:
    params = [param.name for param in method.description.params]
    javaDoc = method.javaDoc  # type: JavaDoc
    javaDoc.variables = ["@variable%d %s" % (i, applyFiltersForString(convert(name).replace(r"_", " "), params)) for i, name in enumerate(params)]
    javaDoc.head = applyFiltersForString(javaDoc.head, params)
    javaDoc.params = [applyFiltersForString(param, params) for param in javaDoc.params]
    javaDoc.results = [applyFiltersForString(result, params) for result in javaDoc.results]
    javaDoc.throws = [applyFiltersForString(throw, params) for throw in javaDoc.throws]
    javaDoc.sees = [applyFiltersForString(see, params) for see in javaDoc.sees]
    method.javaDoc = javaDoc
    return method


def applyFiltersForMethods(methods: list) -> list:
    return [applyFiltersForMethod(method) for method in methods]
