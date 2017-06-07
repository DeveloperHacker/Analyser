import xml.etree.ElementTree
from xml.etree.ElementTree import ElementTree

from utils.method import *
from utils.wrapper import trace


class Tags:
    methods = "methods"
    method = "method"
    java_doc = "java_doc"
    head = "head"
    param = "param"
    result = "return"
    see = "see"
    throws = "throws"
    description = "description"
    name = "name"
    type = "type"
    parameters = "parameters"
    owner = "owner"
    contract = "contract"


@trace
def unpack_methods(path: str) -> list:
    parser: ElementTree = xml.etree.ElementTree.parse(path).getroot()
    methods = []
    for method_tag in parser.findall(Tags.method):
        method = Method()
        method.java_doc = unpack_java_doc(method_tag)
        method.description = unpack_method_description(method_tag)
        method.contract = unpack_contract(method_tag)
        methods.append(method)
    return methods


def unpack_java_doc(parent: ElementTree):
    java_doc = JavaDoc()
    for java_doc_tag in parent.findall(Tags.java_doc):
        for head_tag in java_doc_tag.findall(Tags.head):
            java_doc.head = head_tag.text or ""
        for param_tag in java_doc_tag.findall(Tags.param):
            java_doc.params.append(param_tag.text or "")
        for result_tag in java_doc_tag.findall(Tags.result):
            java_doc.results.append(result_tag.text or "")
        for see_tag in java_doc_tag.findall(Tags.see):
            java_doc.sees.append(see_tag.text or "")
        for throw_tag in java_doc_tag.findall(Tags.throws):
            java_doc.throws.append(throw_tag.text or "")
    return java_doc


def unpack_method_description(parent: ElementTree):
    description = Description()
    for description_tag in parent.findall(Tags.description):
        for name_tag in description_tag.findall(Tags.name):
            description.name = name_tag.text or ""
        for type_tag in description_tag.findall(Tags.type):
            description.type = Type(type_tag.text or "")
        for params_tag in description_tag.findall(Tags.parameters):
            for param_tag in params_tag.findall(Tags.param):
                parameter = Parameter()
                for name_tag in param_tag.findall(Tags.name):
                    parameter.name = name_tag.text or ""
                for type_tag in param_tag.findall(Tags.type):
                    parameter.type = Type(type_tag.text or "")
                    description.params.append(parameter)
        for owner in description_tag.findall(Tags.owner):
            description.owner = Type(owner.text or "")
    return description


def unpack_contract(parent: ElementTree):
    contract = Contract()
    for contract_tag in parent.findall(Tags.contract):
        contract.code = contract_tag.text or ""
    return contract
