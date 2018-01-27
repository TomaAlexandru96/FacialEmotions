#!/usr/bin/env python

import scipy.io as spio


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

    # Parent call to recursive function
    decision_tree_learning(x, attributes, y)


def decision_tree_learning(examples, attributes, binary_targets):
    pass

def check_same_targets(binary_targets):
    pass

def majority_value(binary_targets):
    pass


def choose_best_decision_attribute(examples, attributes, binary_targets):
    pass


if __name__ == "__main__":
    main()
