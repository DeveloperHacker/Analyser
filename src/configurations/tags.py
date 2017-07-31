NORMAL_CONCENTRATION_OF_WORDS = 0.7

STRING = "@string"
NUMBER = "@number"

HTML_BEGIN = "@html_begin"
HTML_END = "@html_end"
HTML_BLOCK = "@html_block"
REFERENCE = "@reference"
LINK = "@link"
PATH = "@path"
URL = "@url"
INVOCATION = "@invocation"
DOT_INVOCATION = "@dot_invocation"
STABLE_REDUCTION = "@stable_reduction"

HEAD = "@head"
PARAMETER = "@param"
RETURN = "@return"
VARIABLE = "@var"
THROW = "@throw"
SEE = "@see"

NEXT = "@next"
GO = "@go"
PAD = "@pad"
NOP = "@nop"

PARTS = (HEAD, PARAMETER, RETURN, VARIABLE, THROW, SEE)

TAGS = PARTS + (
    STRING,
    HTML_BEGIN,
    HTML_END,
    HTML_BLOCK,
    REFERENCE,
    LINK,
    PATH,
    URL,
    INVOCATION,
    DOT_INVOCATION,
    NUMBER,
    STABLE_REDUCTION,

    NEXT,
    GO,
    PAD,
    NOP
)
