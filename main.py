#!/usr/bin/env python

import scipy.io as spio
from tree import TreeNode
import math


def main():
    total_attributes = 45
    number_of_trees = 6

    # Extract data from the file
    mat = spio.loadmat('Data/cleandata_students.mat', squeeze_me=True)
    x = list(mat['x'])
    y = list(mat['y'])

    # Create attributes
    attributes = [0] * 45
    for i in range(total_attributes):
        attributes[i] = i + 1

    trees = []
    # Parent call to recursive function
    for i in range(1, number_of_trees):
        # the binary target for index i
        y_tree = list(map(lambda value: value == i if 1 else 0, y))
        trees.append(decision_tree_learning(x, attributes, y_tree))

    for tree in trees:
        dump_tree(tree)


def decision_tree_learning(examples, attributes, binary_targets):
    if same_binary_targets(binary_targets):
        return TreeNode.create_leaf(binary_targets[0])
    elif len(attributes) == 0:
        return TreeNode.create_leaf(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = TreeNode.create_internal(best_attribute)
        for v in [0, 1]:
            v_examples = []
            v_binary_targets = []
            for i in range(0, len(examples)):
                example = examples[i]
                if example[best_attribute - 1] == v:
                    v_examples.append(example)
                    v_binary_targets.append(binary_targets[i])

            if len(v_examples) == 0:
                return TreeNode.create_leaf(majority_value(binary_targets))
            else:
                attributes.remove(best_attribute)
                subtree = decision_tree_learning(v_examples, attributes, v_binary_targets)
                tree.add_kid(subtree)
                attributes.append(best_attribute)
        return tree


# checks if binary_targets vector contains same values
def same_binary_targets(binary_targets):
    if len(binary_targets) <= 0:
        return True
    first_target = binary_targets[0]
    for target in binary_targets:
        if target != first_target:
            return False
    return True


# finds the mode of the vector
def majority_value(binary_targets):
    return max(set(binary_targets), key=binary_targets.count)


def get_entropy(p, n):
    if p == 0 or n == 0:
        return 0

    return 0

    # TODO
    # return -(p/(p+n)) * math.log(p/(p+n), 2) - (n/(p+n)) * math.log(n/(p+n), 2)


# find between 1 and 45
def choose_best_decision_attribute(examples, attributes, binary_targets):
    max_gain = -1
    index_max = -1

    for attr, index in enumerate(attributes):
        p0 = 0
        n0 = 0
        p1 = 0
        n1 = 0
        p = 0
        n = 0

        for i in range(0, len(examples)):
            example = examples[i]
            if example[attr] == 0:
                if binary_targets[i] == 1:
                    p0 += 1
                    p += 1
                else:
                    n0 += 1
                    n += 1
            else:
                if binary_targets[i] == 1:
                    p1 += 1
                    p += 1
                else:
                    n1 += 1
                    n += 1

        remainder = ((p0 + n0) / (p + n)) * get_entropy(p0, n0) + \
                    ((p1 + n1) / (p + n)) * get_entropy(p1, n1)
        gain = get_entropy(p, n) - remainder
        if gain > max_gain:
            max_gain = gain
            index_max = index

    if index_max == -1:
        raise ValueError("Index is -1")

    return attributes[index_max]


def dump_tree(tree):
    print(tree.to_string())


if __name__ == "__main__":
    main()
