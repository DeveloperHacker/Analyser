HEAD = "@head"
PARAMETER = "@param"
RETURN = "@return"
THROW = "@throw"
SEE = "@see"
SIGNATURE = "@signature"

PARTS = (HEAD, PARAMETER, RETURN, THROW, SEE, SIGNATURE)

STRING = "@string"
NUMBER = "@number"

HTML_BEGIN = "@html"
HTML_END = "@html"
HTML_BLOCK = "@html"
LINK = "@link"
PATH = "@path"
URL = "@url"

UNDEFINED = "@undefined"
GO = "@go"
PAD = "@pad"
NOP = "@nop"
NEXT = "@next"

PARAM_0 = "param[0]"
PARAM_1 = "param[1]"
PARAM_2 = "param[2]"
PARAM_3 = "param[3]"
PARAM_4 = "param[4]"
PARAM_5 = "param[5]"

TAGS = PARTS + (STRING, NUMBER, HTML_BEGIN, HTML_END, HTML_BLOCK, LINK, PATH, URL, GO, PAD, NOP, NEXT)

CONTRACT = "contract"
JAVA_DOC = "java-doc"
DESCRIPTION = "description"

EMBEDDINGS_PATH = 'resources/word2vec/embeddings.pickle'
