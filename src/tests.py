from contract_parser.Parser import Parser

text = "                not_equal(@param[0], @null)\n" + \
       "                not_equal(@param[1], @null)\n" + \
       "                not_equal(@result, @null)\n" + \
       "                strong is(\"The bucket is reset\", @true)\n" + \
       "                strong is(\"The bucket must not be shared\", @true)\n" + \
       "                is(\"parsing is not supported\", @false)\n" + \
       "                is(\"the text to parse is invalid\", @false)\n"

parsed = Parser.parse(text)
print("\n".join("{:15s} {:12s} {}".format(token.__class__.__name__, token.name, word) for token, word in parsed))
tree = Parser.tree(parsed)
print(tree)
flatten = tree.flatten()
print("\n".join("{:15s} {:12s} {}".format(token.__class__.__name__, token.name, word) for token, word in flatten))
