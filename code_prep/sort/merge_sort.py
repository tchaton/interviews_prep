#!/usr/bin/env python

'''
Python Implementation : Merge Sort Algorithm
'''

__author__ = "tchaton"

from ..utils import asssert_func

import numpy as np

def test():
    #arr = np.random.randint(0, 100, (1000,))
    arr = [np.random.randint(0, 100) for _ in range(10)]
    asssert_func(merge_sort, arr, sorted(arr))

def merge_sort(arr):
    def _merge_sort(arr):
        if len(arr) == 1:
            return arr
        if len(arr) == 2:
            if arr[0] <= arr[1]:
                return arr
            else:
                return arr[::-1]
        n = len(arr)//2
        left = merge_sort(arr[:n])
        right = merge_sort(arr[n:])
        merged = []
        cnt_left = 0
        cnt_l = len(left) -1
        cnt_right = 0
        cnt_r = len(right) -1
        while cnt_left <= cnt_l and cnt_right <= cnt_r:
            if left[cnt_left] < right[cnt_right]:
                merged.append(left[cnt_left])
                cnt_left+=1
            else:
                merged.append(right[cnt_right])
                cnt_right+=1
        if cnt_left == cnt_l + 1:
            merged+=right[cnt_right:]
        if cnt_right == cnt_r + 1:
            merged+=left[cnt_left:]
        return merged
    out = _merge_sort(arr)
    return out
