#!/usr/bin/env python


class TreeNode:
    def __init__(self, attribute=None, label=None):
        self.label = label   # Leaf node decision
        self.op = attribute  # Internal node name
        self.kids = []

    def add_kid(self, kid):
        self.kids.append(kid)

    @staticmethod
    def create_leaf(label):
        return TreeNode(label=label)

    @staticmethod
    def create_internal(attribute):
        return TreeNode(attribute=attribute)

    def is_leaf(self):
        return self.label is not None

    def to_string(self):
        return self.__to_string__("")

    def __to_string__(self, prefix):
        if self.is_leaf():
            return "Leaf(" + str(self.label) + ")"
        else:
            sb = "Internal(" + str(self.op) + ")\n"

            for kid in self.kids:
                sb += kid.__to_string__(prefix + "|-") + "\n"

            return sb
