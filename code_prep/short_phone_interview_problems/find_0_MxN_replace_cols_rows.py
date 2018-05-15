#!/usr/bin/env python

'''
Python Implementation : Convert col and row to 0 when a 0 is encountered Algorithm
'''

__author__ = "tchaton"

from ..utils import asssert_func

import numpy as np

def test():
    ''' find 0s in a matrix and replace the col and row to 0s:'''
    m1 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
    asssert_func(find_0_MxN, m1 , [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    m2 = [[1, 2, 3, 4], [0, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    asssert_func(find_0_MxN, m2 , [[0, 2, 3, 4], [0, 0, 0, 0], [0, 10, 11, 12], [0, 14, 15, 16]])
    '''
    Print : 
    before
    [1, 2, 3, 4]
    [5, 6, 7, 8]
    [9, 10, 11, 12]
    [13, 14, 15, 16]

    after
    [1, 2, 3, 4]
    [5, 6, 7, 8]
    [9, 10, 11, 12]
    [13, 14, 15, 16]

    before
    [1, 2, 3, 4]
    [0, 6, 7, 8]
    [9, 10, 11, 12]
    [13, 14, 15, 16]

    after
    [0, 2, 3, 4]
    [0, 0, 0, 0]
    [0, 10, 11, 12]
    [0, 14, 15, 16]
    '''
def find_0_MxN(matrix):
    def _find_0_MxN(matrix):
        n_row = len(matrix)
        n_col = len(matrix[0])
        pos = []
        print('before')
        for i in range(n_row):
            print(matrix[i])
        for i in range(n_row):
            for j in range(n_col):
                if matrix[i][j] == 0:
                    pos.append([i, j])
        for po in pos:
            i, j = po
            matrix[i] = [0 for i in range(n_col)]
            for i in range(n_row):
                matrix[i][j] = 0
        print()
        print('after')
        for i in range(n_row):
            print(matrix[i])
        print()
        return matrix
    matrix = _find_0_MxN(matrix)
    return matrix
