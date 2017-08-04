import itertools
import re
from typing import Iterable

from configurations.tags import *


def anonymize_quoted_strings(string: str) -> str:
    pattern = re.compile(r'"(?:[^"\n\r\\]|(?:"")|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*"' + '|' +
                         r"'(?:[^'\n\r\\]|(?:'')|(?:\\(?:[^x]|x[0-9a-fA-F]+)))*'")
    return re.sub(pattern, " %s " % STRING, string)


def anonymize_tags(string: str) -> str:
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
            is_complete = False
            if not is_end_tag:
                index = len(tags)
                stacks[name].append(index)
            elif len(stacks[name]) > 0:
                index = stacks[name][-1]
                del stacks[name][-1]
                tags[index][1] = True
                is_complete = True
            tags.append([is_end_tag, is_complete, name, begin_index, i])
    line = []
    index = 0
    nesting = {}
    in_block = lambda: any(inst > 0 for name, inst in nesting.items())
    for is_end_tag, is_complete, name, begin_index, end_index in tags:
        if not in_block():
            line.append(string[index:begin_index])
        if is_complete:
            nesting[name] = nesting.get(name, 0) + (-1 if is_end_tag else 1)
            assert nesting[name] >= 0
            if not in_block():
                line.append(" %s " % HTML_BLOCK)
        elif not in_block():
            line.append(" %s " % (HTML_END if is_end_tag else HTML_BEGIN))
        index = end_index + 1
    line.append(" %s " % HTML_BLOCK if in_block() else string[index:])
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
    return re.sub("(\{@|@\{).*\}", " %s " % LINK, string)


def anonymize_paths(string: str) -> str:
    return re.sub(r"(/[a-zA-Z](\w|\.|\\\s)*){2,}", " %s " % PATH, string)


def anonymize_URLs(string: str) -> str:
    pattern = re.compile(r"((\w+:(//|\\\\))?(\w+[\w.@]*\.[a-z]{2,3})(\w|\.|/|\\|\?|=|-)*)|(\w+:(//|\\\\))")
    return re.sub(pattern, " %s " % URL, string)


def anonymize_parameters_names(string: str, params: Iterable[str]) -> str:
    for i, param in enumerate(params):
        code = ord('a') + i
        assert ord('a') <= code <= ord('z')
        name = " " + VARIABLE + chr(code) + " "
        string = re.sub("[^\w]%s[^\w]" % param, name, string)
    return string


def anonymize_numbers(string: str) -> str:
    return re.sub(r"[+-]?\d*([.,]?\d+)+", " %s " % NUMBER, string)


def expand_words_and_symbols(string: str) -> str:
    complex_logic = ('!=', '?=', '==', '<=>', '=>', '<=', '>=')
    complex_op = ('+=', '-=', '&=', '^=', '/=', '*=', '%=', '@=', '--', '++', '?:')
    brackets = ('<', '>', '(', ')', '{', '}', '[', ']')
    operators = ('!', '=', '?', '^', '`', '%', '$', '*', '#', '/', '\\', '&')
    punctuations = ('.', ',', ':', ';')
    literals = itertools.chain(complex_logic, complex_op, brackets, operators, punctuations)
    escaped = ("".join("\\" + ch for ch in literal) for literal in literals)
    pattern = re.compile(r"(%s)" % "|".join(escaped))
    return re.sub(pattern, r" \1 ", string)


def replace_long_spaces(string: str) -> str:
    return re.sub("[\s\t]+", " ", string)


def expand_underscores(string: str):
    return string.replace("_", " ")


def expand_camel_case(string: str):
    return re.sub(r"(\w)([A-Z][a-z0-9])", r"\1 \2", string)


def apply(string: str, params: Iterable[str]) -> str:
    string = " %s " % string
    string = anonymize_parameters_names(string, params)
    string = anonymize_numbers(string)
    string = anonymize_tags(string)
    string = anonymize_quoted_strings(string)
    string = expand_html_escapes(string)
    string = anonymize_links(string)
    string = anonymize_URLs(string)
    string = anonymize_paths(string)
    string = expand_underscores(string)
    string = expand_camel_case(string)
    string = expand_words_and_symbols(string)
    string = replace_long_spaces(string)
    return string.strip().lower()
