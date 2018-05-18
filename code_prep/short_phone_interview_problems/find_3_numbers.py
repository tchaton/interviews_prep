#!/usr/bin/env python

'''
Python Implementation : Find 3 numbers that sum to zero
'''

__author__ = "tchaton"

from ..utils import asssert_func

def test():
    assert find_3_number([1, 2, -3, 5, -1, -4, 50]) == [[-3, 1, 2], [-1, -1, 2], [-4, -1, 5], [-4, 2, 2]]
    assert find_3_number([]) == []

def find_3_number(arr):
    d = {}
    solutions = []
    for index, i in enumerate(arr):
        d[str(i)] = index
    keys = d.keys()
    for index, x in enumerate(arr):
        sub_arr = arr[:index-1]+arr[index+1:]
        for index2, y in enumerate(sub_arr):
            key = str(-(x+y))
            if key in keys:
                if d[key] != index:
                    sol = sorted([x, y, int(key)])
                    if sol not in solutions:
                        solutions.append(sol)
    return solutions
