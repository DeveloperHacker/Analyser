
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

class Contract:

    def __init__(self):
        self.enters = []
        self.exits = []
        self.exitIds = []

    def __str__(self) -> str:
        lines = [":::ENTER"]
        for enter in self.enters:
            lines.append(enter)
        lines.append(":::EXIT")
        for _exit in self.exits:
            lines.append(_exit)
        for exitId in self.exitIds:
            lines.append(":::EXIT-{}".format(exitId["id"]))
            for _exit in exitId["exits"]:
                lines.append(_exit)
        return "\n".join(lines)

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
        self.javaDoc = JavaDoc()
        self.contract = Contract()

    def __str__(self) -> str:
        return "{}\n{}\n{}".format(str(self.javaDoc), str(self.description), str(self.contract))
