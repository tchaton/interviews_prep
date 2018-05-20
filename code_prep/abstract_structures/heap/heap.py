#!/usr/bin/env python

__author__ = "tchaton"

import numpy as np

'''
Python Implementation Heap
'''

def test():
    heap = Heap()
    arr = np.random.randint(0, 100, (100,))
    for i in arr:
        heap.add(i)

    assert sorted(heap.heap) == sorted(arr) # ALL ELEMENTS ARE INSIDE THE HEAP

class Heap:

    def __init__(self):
        self.h = []

    def add(self, value):
        self.h.append(value)
        if not (len(self.h) <= 1):
            self._heapify()

    def _heapify(self):
        n = len(self.h) - 1
        last_v = self.h[n]
        while self.h[n // 2] < last_v:
            self.h[n] = self.h[n // 2]
            n = n // 2
            self.h[n] = last_v
        self.h[n] = last_v

    def print_heap(self):
        print(self.h)

    @property
    def heap(self):
        return self.h
