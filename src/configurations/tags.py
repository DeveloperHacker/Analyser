HEAD = "@head"
PARAMETER = "@param"
RETURN = "@return"
VARIABLE = "@var"
THROW = "@throw"
SEE = "@see"

PARTS = (HEAD, PARAMETER, RETURN, VARIABLE, THROW, SEE)

STRING = "@string"
NUMBER = "@number"

HTML_BEGIN = "@html"
HTML_END = "@html"
HTML_BLOCK = "@html"
LINK = "@link"
PATH = "@path"
URL = "@url"

GO = "@go"
PAD = "@pad"
NOP = "@nop"
NEXT = "@next"

TAGS = PARTS + (STRING, NUMBER, HTML_BEGIN, HTML_END, HTML_BLOCK, LINK, PATH, URL, GO, PAD, NOP, NEXT)
