#!/usr/bin/env python

'''
Python Implementation : Bubble Sort Algorithm
'''

__author__ = "tchaton"

from ..utils import asssert_func

import numpy as np

def test():
    arr = [np.random.randint(0, 100) for _ in range(10)]
    asssert_func(buddle_sort, arr, sorted(arr))

def buddle_sort(arr):
    def _buddle_sort(arr):
        copy_arr = arr[::]
        L = len(arr)
        for i in range(len(arr) -1):
            pivot_index = 0
            pivot = copy_arr[0]
            if i == 0:
                list_arr = copy_arr
            else:
                list_arr = copy_arr[:-i]
            for index, x in enumerate(list_arr):
                if index == 0:
                    pass
                else:
                    if x < pivot:
                        copy_arr[pivot_index], copy_arr[index] = copy_arr[index], copy_arr[pivot_index]
                        pivot_index = index
                    else:
                        pivot_index = index
                        pivot = x
        return copy_arr
    out = _buddle_sort(arr)
    return out
