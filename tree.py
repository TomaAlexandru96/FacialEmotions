#!/usr/bin/env python

class TreeNode:
    def __init__(self, label):
        self.label = label

    def __init__(self, attribute):
        self.op = attribute
        self.kids = []

    def add_kid(self, kid):
        self.kids.append(kid)
