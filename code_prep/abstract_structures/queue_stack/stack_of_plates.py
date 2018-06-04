# Implement a class that acts as a single stack made out of multiple stacks
# which each have a set capacity.

__author__ = 'tchaton'

class MultiStack():
    def __init__(self, cap):
        self.cap = cap
        self.stacks = []

    def push(self, item):
        if len(self.stacks) and (len(self.stacks[-1]) < self.cap):
            self.stacks[-1].append(item)
        else:
            self.stacks.append([item])

    def pop(self):
        if len(self.stacks) > 0:
            if len(self.stacks[-1]) > 0:
                return self.stacks[-1].pop()
            else:
                self._refound()
                if len(self.stacks[-1]) > 0:
                    return self.stacks[-1].pop()
                else:
                    return None
        else:
            return None

    def _refound(self):
        arr = []
        h = []
        for stack in self.stacks:
            for v in stack:
                h.append(v)
                if len(h) == self.cap:
                    arr.append(h)
                    h = []
        self.stacks = arr
        if len(self.stacks) == 0:
            self.stacks = [[]]

    def pop_at(self, value):
        if len(self.stacks) >= value:
            if len(self.stacks[value]) > 0:
                return self.stacks[value].pop()
            else:
                self._refound()
                return None
        else:
            print('Can t access an un-existing column')

    def _print(self):
        print(self.stacks)

import unittest

class Test(unittest.TestCase):
    def test_multi_stack(self):
        stack = MultiStack(3)
        stack.push(11)
        stack.push(22)
        stack.push(33)
        stack.push(44)
        stack.push(55)
        stack.push(66)
        stack.push(77)
        stack.push(88)
        stack._print()
        self.assertEqual(stack.pop(), 88)
        stack._print()
        self.assertEqual(stack.pop_at(1), 66)
        stack._print()
        self.assertEqual(stack.pop_at(0), 33)
        stack._print()
        self.assertEqual(stack.pop_at(1), 55)
        stack._print()
        self.assertEqual(stack.pop_at(1), 44)
        stack._print()
        self.assertEqual(stack.pop_at(1), None)
        stack._print()
        stack.push(99)
        stack._print()
        self.assertEqual(stack.pop(), 99)
        stack._print()
        self.assertEqual(stack.pop(), 77)
        stack._print()
        self.assertEqual(stack.pop(), 22)
        stack._print()
        self.assertEqual(stack.pop(), 11)
        stack._print()
        self.assertEqual(stack.pop(), None)

if __name__ == "__main__":
    unittest.main()
