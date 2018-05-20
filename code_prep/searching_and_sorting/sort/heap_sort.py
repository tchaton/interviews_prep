#!/usr/bin/env python

__author__ = "tchaton"

from ..heap.heap import Heap

def test():
    seq = [3, 5, 2, 6, 8, 1, 0, 3, 5, 6, 2, 5, 4, 1, 5, 3]
    assert(heap_sort(seq) == sorted(seq))


def heap_sort(arr):
    '''
    Best : O(nln(n))
    Avg : O(nln(n))
    Worst : O(nln(n))
    Reference : https://en.wikipedia.org/wiki/Heapsort
    '''
    def _heap_sort(arr):
        if len(arr) == 0:
            return []
        heap = Heap()
        for x in arr:
            heap.add(x)
        return  _heap_sort(heap.heap[1:]) + [heap.heap[0]]

    return _heap_sort(arr)
