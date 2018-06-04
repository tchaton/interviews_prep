# Implement a stack with a function that returns the current minimum item.

__author__ = 'tchaton'
import unittest
import numpy as np
from collections import defaultdict
import numpy as np

class Node:

    def __init__(self, value, prev=None, next=None, cnt=1):
        self.value = value
        self.prev = prev
        self.next = next
        self.cnt = cnt

class StackwMin:

    def __init__(self):
        self.arr = []
        self.tail = Node(np.inf)

    def push(self, item):
        self.arr.append(item)
        if item < self.tail.value:
            n = Node(item, prev=self.tail)
            self.tail = n

    def pop(self):
        last_value = self.arr[-1]
        self.arr = self.arr[:-1]
        if last_value <= self.tail.value:
            self.tail = self.tail.prev
        return last_value

    def min(self):
        return self.tail.value

class Test(unittest.TestCase):
    def test_min_stack(self):
        min_stack = StackwMin()
        self.assertEqual(min_stack.min(), np.inf)
        min_stack.push(7)
        self.assertEqual(min_stack.min(), 7)
        min_stack.push(6)
        min_stack.push(5)
        self.assertEqual(min_stack.min(), 5)
        min_stack.push(10)
        self.assertEqual(min_stack.min(), 5)
        self.assertEqual(min_stack.pop(), 10)
        self.assertEqual(min_stack.pop(), 5)
        self.assertEqual(min_stack.min(), 6)
        self.assertEqual(min_stack.pop(), 6)
        self.assertEqual(min_stack.pop(), 7)
        self.assertEqual(min_stack.min(), np.inf)

if __name__ == "__main__":
    unittest.main()
