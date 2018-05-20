#!/usr/bin/env python

__author__ = "tchaton"


''' Find the mth-to-last element of a linked list.
    One option is having two pointers, separated by m. P1 start at the roots
    (p1 = self.root) and p2 is m-behinf pointer, which is created when p1 is at m.
    When p1 reach the end, p2 is the node. '''

from .base import Node, LinkedList

def test():
    N = 10
    k = 5
    llk = LinkedList_find_kth()
    for i in range(10):
        llk.add(i)

    node_k = llk.find_kth_to_last(k)
    assert node_k.value == (N - k) - 1


class LinkedList_find_kth(LinkedList):

    def find_kth_to_last(self, k):
        node = self.head
        cnt = 0
        node_k = self.head
        try:
            while node.next != None:
                node = node.next
                cnt+=1
                if cnt > k:
                    node_k = node_k.next
            return node_k
        except:
            Exception('The list doesn t contain k elements')
