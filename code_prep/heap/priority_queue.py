#!/usr/bin/env python

__author__ = "tchaton"

from ..heap.heap import Heap
import numpy as np

def test():
    ''' push and pop are all O(logN) '''
    q = PriorityQueue()
    q.push(Item('test1', 1))
    q.push(Item('test2', 4))
    q.push(Item('test3', 2))
    #q._print()
    assert(str(q.pop()) == "Item('test2')")
    assert(str(q.pop()) == "Item('test3')")
    assert(str(q.pop()) == "Item('test1')")

class HeapItem(Heap):

    def __init__(self):
        super(HeapItem, self).__init__()

    def _heapify(self):
        n = len(self.h) - 1
        last_v = self.h[n]
        while self.h[n // 2].value < last_v.value:
            self.h[n] = self.h[n // 2]
            n = n // 2
            self.h[n] = last_v
        self.h[n] = last_v

    def _print(self):
        print([h.value for h in self.h])

class Item:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "Item({!r})".format(str(self.name))

class PriorityQueue(object):

    def __init__(self):
        self.heapItem = HeapItem()
        self.index = 0

    def push(self, item):
        self.heapItem.add(item)

    def pop(self):
        first = self.heapItem.h[0]
        self.heapItem.h = self.heapItem.h[1:]
        return first

    def _print(self):
        self.heapItem._print()
