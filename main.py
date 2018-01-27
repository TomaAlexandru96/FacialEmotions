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
    attributes = []
    for i in range(total_attributes):
        attributes.append(i+1)
    y = [1,1,2,2,2,2,4,4,4,4,4,4]
    # Parent call to recursive function
    decision_tree_learning(x, attributes, y)


def decision_tree_learning(examples, attributes, binary_targets):
    if same_binary_targets(binary_targets):
        return TreeNode(binary_targets[0])
    elif len(attributes) == 0:
        return TreeNode(majority_value(binary_targets))
    else:
        best_attribute = choose_best_decision_attribute(examples, attributes, binary_targets)
        # root_attribute = TreeNode()

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
    pass


if __name__ == "__main__":
    main()
