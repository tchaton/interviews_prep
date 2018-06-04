# Return an intersecting node if two linked lists intersect.

__author__ = 'tchaton'
import unittest
from base import LinkedList, Node
import numpy as np
from collections import defaultdict

def get_key(d, key):
    try:
        return d[key]
    except:
        return None

def intersection(lk1, lk2):
    if lk1.head == None:
        return None
    if lk2.head == None:
        return None
    n1 = lk1.head
    n2 = lk2.head
    nodes = defaultdict(list)
    while n1:
        nodes[n1.value].append(n1)
        n1 = n1.next
    intersections = []
    while n2:
        n_list = get_key(nodes, n2.value)
        if len(n_list) > 0:
            for n in n_list:
                copy_n = n2
                intersection = []
                while copy_n.value == n.value:
                    intersection.append(copy_n.value)
                    try:
                        copy_n = copy_n.next
                        n = n.next
                        if n == None:
                            break
                        if copy_n == None:
                            break
                    except:
                        break
            if len(intersection) > 1:
                intersections.append(intersection)
            n2 = n2.next
        else:
            n2 = n2.next
    return intersections

class Test(unittest.TestCase):
    def test_intersection(self):
        arr = np.array([1, 0, 0, 5, 6, 7, 8, 9])
        arr2 = np.array([1, 0, 1, 5, 6, 6, 8, 9])
        lk1 = LinkedList()
        lk2 = LinkedList()
        for x, y in zip(arr, arr2):
            lk1.add(x)
            lk2.add(y)
        self.assertEqual(intersection(lk1, lk2), [[1, 0], [5, 6], [8, 9]])
        arr = np.array([1, 5, 6, 5, 6, 7, 8, 9])
        arr2 = -np.array([1, 8, 1, 5, 6, 6, 8, 9])
        lk1 = LinkedList()
        lk2 = LinkedList()
        for x, y in zip(arr, arr2):
            lk1.add(x)
            lk2.add(y)
        self.assertEqual(intersection(lk1, lk2), [])

if __name__ == "__main__":
    unittest.main()
