import re


def filterTags(string: str) -> str:
    return re.sub("\<.*\>", "@tag", string)


def filterLinks(string: str) -> str:
    pass


def filterParamNames(string: str, params: list) -> str:
    pass
