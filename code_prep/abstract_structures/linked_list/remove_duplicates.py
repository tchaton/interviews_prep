# Remove the duplicate values from a linked list.

__author__ = 'tchaton'
import unittest
from base import Node, LinkedList

def remove_duplicates(lk):
    seen = []
    def _remove_duplicates(node, node_prev=None):
        if node is not None:
            value = node.value
            if value not in seen:
                seen.append(value)
                _remove_duplicates(node.next, node_prev=node)
            else:
                node_prev.next = node.next
                _remove_duplicates(node_prev.next, node_prev=node_prev)

    _remove_duplicates(lk.head, node_prev=None)


class Test(unittest.TestCase):
    def test_remove_duplicates(self):
        arr = [1, 2, 1, 2, 5, 6, 7, 5, 9]
        arr2 = list(set([1, 2, 1, 2, 5, 6, 7, 5, 9]))
        lk = LinkedList()
        for v in arr:
            lk.add(v)
        self.assertEqual(lk._printList(), arr)
        remove_duplicates(lk)
        self.assertEqual(lk._printList(), arr2)

if __name__ == "__main__":
    unittest.main()
