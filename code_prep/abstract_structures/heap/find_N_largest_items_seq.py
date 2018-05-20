#!/usr/bin/env python

__author__ = "tchaton"

from ..heap.heap import Heap
import numpy as np

def test():
    N = 5
    seq = [3, 5, 2, 6, 8, 1, 0, 3, 5, 6, 2, 5, 4, 1, 5, 3]
    heap_N_largest = find_N_largest_items_seq(seq, N)
    numpy_N_largest = np.sort(seq)[-N:]
    assert sorted(heap_N_largest) == sorted(numpy_N_largest)

def find_N_largest_items_seq(arr, N):
    heap = Heap()
    for x in arr:
        heap.add(x)
    return np.array(heap.heap[:N][::-1])
