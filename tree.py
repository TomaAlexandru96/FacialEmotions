import random

class TreeNode:
    def __init__(self, attribute=None, label=None, entropy=None):
        self.label = label   # Leaf node decision
        self.op = attribute  # Internal node name
        self.kids = []
        self.entropy = entropy

    def add_kid(self, kid):
        self.kids.append(kid)
        return self

    @staticmethod
    def create_leaf(label, entropy):
        return TreeNode(label=label, entropy=entropy)

    @staticmethod
    def create_internal(attribute):
        return TreeNode(attribute=attribute)

    def is_leaf(self):
        return self.label is not None

    def parse_tree(self, data, tree_height):
        if self.is_leaf():
            return self.label, self.entropy, tree_height
        else:
            return self.kids[data[self.op - 1]].parse_tree(data, tree_height + 1)

    def get_value(self, val):
        if self.is_leaf():
            if self.label == True:
                return val
            else:
                return -val
        else:
            ret = 0
            for i in range(len(self.kids)):
                ret += self.kids[i].get_value(val/2)
            return ret

    def prune_tree(self, tree_height, prob):
        if tree_height < 5:
            for i in range(len(self.kids)):
                self.kids[i].prune_tree(tree_height+1, prob)
        else:
            for i in range(len(self.kids)):
                rand = random.randint(1,100)
                if rand <= prob:
                    val = self.kids[i].get_value(1)
                    if val > 0:
                        self.kids[i] = TreeNode.create_leaf(True, 0.5)
                    else:
                        self.kids[i] = TreeNode.create_leaf(False, 0.5)
                else:
                    self.kids[i].prune_tree(tree_height+1,prob)

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
