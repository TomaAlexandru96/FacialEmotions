#!/usr/bin/env python

import scipy.io as spio
from tree import TreeNode


def main():
    total_attributes = 45

    # Extract data from the file
    mat = spio.loadmat('Data/cleandata_students.mat', squeeze_me=True)
    x = mat['x']
    y = mat['y']

    # Create attributes
    attributes = [0]*45
    for i in range(total_attributes):
        attributes[i] = i+1

    # Parent call to recursive function
    decision_tree_learning(x, attributes, y)


def decision_tree_learning(examples, attributes, binary_targets):
    if same_binary_targets(binary_targets):
        return TreeNode(binary_targets[0])
    elif len(attributes) == 0:
        return TreeNode(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        tree = TreeNode(best_attribute)
        for v in [0,1]:
            v_examples = []
            v_binary_targets = []
            for i in range(0, len(examples)):
                example = examples[i]
                if example[best_attribute - 1] == v:
                    v_examples.append(example)
                    v_binary_targets.append(binary_targets[i])
            if len(v_examples) == 0:
                return TreeNode(majority_value(binary_targets))
            else:
                attributes.remove(best_attribute)
                subtree = decision_tree_learning(v_examples, attributes, v_binary_targets)
                tree.add_kid(subtree)
                attributes.append(best_attribute)
        return tree


def same_binary_targets(binary_targets):
    if len(binary_targets) <= 0:
        return True
    first_target = binary_targets[0]
    for target in binary_targets:
        if target != first_target:
            return False
    return True


def majority_value(binary_targets):
    return max(set(binary_targets), key=binary_targets.count)


def choose_best_decision_attribute(examples, attributes, binary_targets):
    return attributes[0]


if __name__ == "__main__":
    main()
