class Type:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class Parameter:
    def __init__(self):
        self.name = ""
        self.type = Type("")

    def __str__(self) -> str:
        return "{} {}".format(self.type, self.name)


class JavaDoc:
    def __init__(self):
        self.head = ""
        self.params = []
        self.variables = []
        self.results = []
        self.sees = []
        self.throws = []

    def __str__(self) -> str:
        tmp = [self.head]
        tmp.extend(self.params)
        tmp.extend(self.variables)
        tmp.extend(self.results)
        tmp.extend(self.sees)
        tmp.extend(self.throws)
        return "/**\n * {}\n **/".format("\n * ".join(tmp))

    def empty(self) -> bool:
        return len(self.head) == 0 and \
               len(self.params) == 0 and \
               len(self.results) == 0 and \
               len(self.sees) == 0 and \
               len(self.throws) == 0


class Contract:
    def __init__(self):
        self.code = ""

    def __str__(self) -> str:
        return self.code

    def empty(self) -> bool:
        return self.code is ""


class Description:
    def __init__(self):
        self.name = ""
        self.type = Type("")
        self.params = []
        self.owner = Type("")

    def __str__(self) -> str:
        name = self.name
        tyre = str(self.type)
        owner = str(self.owner)
        params = ", ".join([str(param) for param in self.params])
        return "{} {}.{}({})".format(tyre, owner, name, params)


class Method:
    def __init__(self):
        self.description = Description()
        self.java_doc = JavaDoc()
        self.contract = Contract()

    def __str__(self) -> str:
        return "{}\n{}\n{}".format(str(self.java_doc), str(self.description), str(self.contract))
