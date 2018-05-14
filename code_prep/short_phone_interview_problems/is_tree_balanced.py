#!/usr/bin/env python

'''
Python Implementation : is_tree_balanced
'''

__author__ = "tchaton"

from .tree import Tree
from .queue import Queue

def test():
    tree = Tree()
    l = [10, 5, 6, 3, 8, 2, 1, 11, 9, 4]
    for i in l:
        tree.add(i)
    assert evaluate_balanced(max_depth(tree)) == False

def aggregate(max_depth_left, max_depth_right):
    cnd_left = isinstance(max_depth_left, tuple)
    cnd_right = isinstance(max_depth_right, tuple)
    if cnd_left and not cnd_right:
        a, b = max_depth_left
        max_depth_left = a
        max_depth_right = max(max_depth_right, b)
    if not cnd_left and cnd_right:
        c, d = max_depth_right
        max_depth_right = d
        max_depth_left = max(max_depth_left, c)
    if cnd_left and cnd_right:
        a, b = max_depth_left
        c, d = max_depth_right
        max_depth_left = max(a, c)
        max_depth_right = max(b, d)
    return max_depth_left, max_depth_right

def evaluate_balanced(mm_v):
    if abs(mm_v[0] - mm_v[1]) < 2:
        return True
    else:
        return False

def max_depth(tree):
    root = tree.root
    def _max_depth(node, max_depth_left=0, max_depth_right=0):
        if node.left:
            max_depth_left = _max_depth(node.left,
                                     max_depth_left=max_depth_left+1,
                                     max_depth_right=max_depth_right)
        if node.right:
            max_depth_right = _max_depth(node.right,
                                             max_depth_left=max_depth_left,
                                             max_depth_right=max_depth_right+1)
        return aggregate(max_depth_left, max_depth_right)
    max_depth_left = max(_max_depth(root.left, max_depth_left=0, max_depth_right=0))+1
    max_depth_right = max(_max_depth(root.right, max_depth_left=0, max_depth_right=0))+1
    return [max_depth_left, max_depth_right]
