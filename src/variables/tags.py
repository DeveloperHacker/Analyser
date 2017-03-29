MAX_NUM_PARAMS = 10
MAX_NUM_NUMBER = 2

STRING = "@string"
VARIABLE = "@variable"
NUMBER = "@number"
TRUE = "@true"
FALSE = "@false"
NULL = "@null"

HTML = "@html"
REFERENCE = "@reference"
LINK = "@link"
PATH = "@path"
URL = "@url"
INVOCATION = "@invocation"
DOT_INVOCATION = "@dotInvocation"
STABLE_REDUCTION = "@sr"

HEAD = "@head"
PARAMS = "@param"
RETURN = "@return"
VARIABLES = "@variable"
SEE = "@see"
THROW = "@throw"

NEXT = "@next"

GO = "@go"
PAD = "@pad"

NORMAL_CONCENTRATION_OF_WORDS = 0.7

PARTS = (
    HEAD[1:],
    PARAMS[1:],
    VARIABLES[1:],
    RETURN[1:],
    # SEE[1:],
    # THROW[1:]
)

TAGS = (
    STRING,
    HTML,
    REFERENCE,
    LINK,
    PATH,
    URL,
    VARIABLE,
    INVOCATION,
    DOT_INVOCATION,
    NUMBER,
    STABLE_REDUCTION,
    HEAD,
    PARAMS,
    RETURN,
    THROW,
    SEE,
    NEXT
)

PUNCTUATION = (".", ",", ":", ";", "(", ")", "{", "}")
