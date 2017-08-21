import re

from contants import *


def anonymize_quoted_strings(string: str) -> str:
    pattern = re.compile(r'"(?:[^"\n\r\\]|(?:"")|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*"' + '|' +
                         r"'(?:[^'\n\r\\]|(?:'')|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*'")
    return re.sub(pattern, " %s " % STRING, string)


def anonymize_tags(string: str) -> str:
    def skip():
        result = pair is None
        result = result or name in skip_tags
        result = result or is_end_tag and string[pair[4] + 1:begin_index].isalpha()
        result = result or not is_end_tag and string[end_index + 1:pair[3]].isalpha()
        return result

    skip_tags = ("p", "b", "i", "u")
    tags = []
    stacks = {}
    begin_index = None
    for i, ch in enumerate(string):
        if ch == "<":
            begin_index = i
        elif ch == ">" and begin_index is not None:
            tag = string[begin_index:i + 1]
            is_end_tag = tag[:2] == "</"
            name = tag[2:-1] if is_end_tag else tag[1:-1]
            if name not in stacks:
                stacks[name] = []
            tag = [is_end_tag, None, name, begin_index, i]
            if not is_end_tag:
                index = len(tags)
                stacks[name].append(index)
            elif len(stacks[name]) > 0:
                index = stacks[name][-1]
                del stacks[name][-1]
                tags[index][1] = tag
                tag[1] = tags[index]
            tags.append(tag)
    line = []
    index = 0
    nesting = {}
    in_block = lambda: any(inst > 0 for name, inst in nesting.items())
    for is_end_tag, pair, name, begin_index, end_index in tags:
        if not in_block():
            line.append(string[index:begin_index])
        if not skip():
            nesting[name] = nesting.get(name, 0) + (-1 if is_end_tag else 1)
            if not in_block():
                line.append(" %s " % HTML_BLOCK)
        index = end_index + 1
    line.append(" %s " % HTML_BLOCK if in_block() and not skip() else string[index:])
    return "".join(line)


def expand_html_escapes(string: str) -> str:
    string = string.replace("&lt;", "<")
    string = string.replace("&gt;", ">")
    string = string.replace("&amp;", "&")
    string = string.replace("&quot;", "\"")
    string = string.replace("&ensp;", " ")
    string = string.replace("&emsp;", " ")
    string = string.replace("&thinsp;", " ")
    string = string.replace("&nbsp;", " ")
    return string


def anonymize_links(string: str) -> str:
    # noinspection Annotator,Annotator
    return re.sub("(\{@|@\{).*?\}", " %s " % LINK, string)


def anonymize_paths(string: str) -> str:
    return re.sub(r"(/[a-zA-Z](\w|\.|\\\s)*){2,}", " %s " % PATH, string)


def anonymize_URLs(string: str) -> str:
    pattern = re.compile(r"((\w+:(//|\\\\))?(\w+[\w.@]*\.[a-z]{2,3})(\w|\.|/|\\|\?|=|-)*)|(\w+:(//|\\\\))")
    return re.sub(pattern, " %s " % URL, string)


def anonymize_numbers(string: str) -> str:
    return re.sub(r"[+-]?\d*([.,]?\d+)+", " %s " % NUMBER, string)


def expand_words_and_symbols(string: str) -> str:
    complex_logic = ('!=', '?=', '==', '<=>', '=>', '<=', '>=', "->", "<-", "<->", "===")
    complex_op = ('+=', '-=', '&=', '^=', '/=', '*=', '%=', '@=', '--', '++', '?:', ":-", "::-", ":=", "::=")
    brackets = ('<', '>', '(', ')', '{', '}', '[', ']')
    operators = ('!', '=', '?', '^', '`', '%', '$', '*', '#', '/', '\\', '&')
    punctuations = ('.', ',', ':', ';')
    quotes = ("'", '"')
    literals = complex_logic + complex_op + brackets + operators + punctuations + quotes
    escaped = ("".join("\\" + ch for ch in literal) for literal in literals)
    pattern = re.compile(r"(%s)" % "|".join(escaped))
    return re.sub(pattern, r" \1 ", string)


def replace_long_spaces(string: str) -> str:
    return re.sub("[\s\t]+", " ", string)


def expand_underscores(string: str):
    return string.replace("_", " ")


def expand_camel_case(string: str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)


def apply(string: str) -> str:
    string = " %s " % string
    string = anonymize_numbers(string)
    string = anonymize_tags(string)
    string = expand_html_escapes(string)
    string = anonymize_links(string)
    string = anonymize_URLs(string)
    string = anonymize_paths(string)
    string = expand_underscores(string)
    string = expand_camel_case(string)
    string = expand_words_and_symbols(string)
    string = replace_long_spaces(string)
    string = string.strip().lower()
    return string
