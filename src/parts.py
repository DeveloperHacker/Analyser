
class Type:

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class JavaDoc:

    def __init__(self):
        self.head = ""
        self.params = []
        self.results = []
        self.sees = []
        self.throws = []

    def __str__(self) -> str:
        tmp = [self.head]
        tmp.extend(self.params)
        tmp.extend(self.results)
        tmp.extend(self.sees)
        tmp.extend(self.throws)
        return "/**\n{}\n */".format("\n".join(tmp))


class Method:

    def __init__(self):
        self.name = ""
        self.type = Type("")
        self.params = []
        self.javaDoc = JavaDoc()
        self.owner = Type("")

    def __str__(self) -> str:
        javaDoc = str(self.javaDoc)
        tyre = str(self.type)
        owner = str(self.owner)
        params = ", ".join([str(param) for param in self.params])
        return "{}\n{} {}.{}({})".format(javaDoc, tyre, owner, self.name, params)


class Parameter:

    def __init__(self):
        self.name = ""
        self.type = Type("")

    def __str__(self) -> str:
        return "{} {}".format(self.type, self.name)
