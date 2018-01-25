#!/usr/bin/env python

import scipy.io as spio


def main():
    mat = spio.loadmat('Data/cleandata_students.mat', squeeze_me=True)
    x = mat['x']
    y = mat['y']
    print (x.size)
    print (y)


def decision_tree_learning(examples, attributes, binary_targets):
    pass


def majority_value(binary_targets):
    pass


def choose_best_decision_attribute(examples, attributes, binary_targets):
    pass


if __name__ == "__main__":
    main()
