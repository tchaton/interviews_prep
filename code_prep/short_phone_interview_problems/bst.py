#!/usr/bin/env python

'''
Python Implementation : Breadth First Search
'''

__author__ = "tchaton"

from .tree import Tree
from .queue import Queue

def test():
    tree = Tree()
    l = [10, 5, 6, 3, 8, 2, 1, 11, 9, 4]
    for i in l:
        tree.add(i)
    assert bst(tree) == [10, 5, 11, 3, 6, 2, 4, 8, 1, 9]


def bst(tree):
    root = tree.root
    nodes = []
    queue = Queue()
    queue.enqueue(root)

    while queue.size() != 0:
        node = queue.peak()
        nodes.append(node.value)
        if node.left:
            queue.enqueue(node.left)
        if node.right:
            queue.enqueue(node.right)
    return nodes
