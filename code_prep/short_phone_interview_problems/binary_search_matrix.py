#!/usr/bin/env python

'''
Python Implementation : Binary Search Matrix
'''

__author__ = "tchaton"

'''TO FINISH'''

import numpy as np

def test():
    matrix = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert binary_search_matrix(matrix.tolist(), 1) == (0, 1)
    assert binary_search_matrix(matrix.tolist(), -1) == None
    assert binary_search_matrix(matrix.tolist(), 8) == (2, 2)

def binary_search_matrix(m, value):
    rows = len(m)
    cols = len(m[0])
    def _binary_search_matrix(m, key, lo=0, hi=None):
        if not m or hi <= lo:
            return None

        mid = (hi + lo)//2
        row = mid//cols
        col = mid%cols
        item = m[row][col]

        if key == item:
            return row, col
        elif key < item:
            return _binary_search_matrix(m, key, lo, mid-1)
        else:
            return _binary_search_matrix(m, key, mid+1, hi)

    return _binary_search_matrix(m, value, hi=rows*cols)
