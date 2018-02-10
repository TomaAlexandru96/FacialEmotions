import pickle

class TreeNode:
    def __init__(self, attribute=None, label=None):
        self.label = label   # Leaf node decision
        self.op = attribute  # Internal node name
        self.kids = []

    def add_kid(self, kid):
        self.kids.append(kid)
        return self

    @staticmethod
    def create_leaf(label):
        return TreeNode(label=label)

    @staticmethod
    def create_internal(attribute):
        return TreeNode(attribute=attribute)

    def is_leaf(self):
        return self.label is not None

    def parse_tree(self, data):
        if self.is_leaf():
            return self.label
        else:
            return self.kids[data[self.op - 1]].parse_tree(data)

    def to_string(self):
        return self.__to_string__("", True, False)

    def __to_string__(self, prefix, is_root, is_last_kid):
        line = ""
        if not is_root:
            line = "-"

        prefix2 = prefix
        if is_last_kid:
            prefix2 = prefix[:-1]
            prefix2 += "|"

        if self.is_leaf():
            return prefix2 + line + "Leaf(" + str(self.label) + ")"
        else:
            sb = prefix2 + line + "Internal(" + str(self.op) + "): " + str(len(self.kids))

            if not is_root:
                prefix += " "

            for index, kid in enumerate(self.kids):
                if index == len(self.kids) - 1:
                    p = prefix + " "
                else:
                    p = prefix + "|"
                sb += "\n" + kid.__to_string__(p, False, index == len(self.kids) - 1)

            return sb

    def save_tree(self, name):
        pickle.dump(self, open(str(name) + ".p", "wb"))

    @staticmethod
    def load_tree(name):
        return pickle.load(open(name + ".p", "rb"))
