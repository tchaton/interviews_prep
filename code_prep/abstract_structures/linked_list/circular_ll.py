#!/usr/bin/env python

__author__ = "tchaton"


''' implement a function to see whether a linked list is circular.
    To implement this, we just need two pointers with different
    paces (for example, one goes twice faster)'''

from .base import Node, LinkedList

class CircularLinkedNode(LinkedList):

    def __init__(self, start=0, modulo=None):
        self.start = start
        self.modulo = modulo
        self.cnt = 0
        self.stop_criteria = False
        super(CircularLinkedNode, self).__init__()

    def add(self, value):
        if not self.stop_criteria:
            if self.head is None:
                self.head = Node(value)
            else:
                if self.cnt >= (self.modulo + self.start):
                    node_circular = self.getNode(self.start)
                    self.head._add(Node(value, next=node_circular))
                    self.stop_criteria = True
                else:
                    self.head._add(Node(value))
                    self.cnt+=1

    def getNode(self, index):
        if index == 0:
            return self.head
        else:
            node = self.head
            try:
                for _ in range(index):
                    node = node.next
                return node
            except:
                Exception('This index can t be found')

    def getLastNode(self):
        node = self.head
        while isinstance(node, Node):
            node = node.next
        return node


def step(node, step=1):
    for _ in range(step):
        node = node.next
    return node

def is_circular(linked_list):
    def _is_circular(start_node):
        l1 = step(start_node, step=1)
        l2 = step(start_node, step=2)
        while l1 != l2:
            l1 = step(l1, step=1)
            l2 = step(l2, step=2)
        return True

    if linked_list is None:
        return False
    else:
        head = linked_list.head
        if head is None:
            return False
        else:
            try:
                return _is_circular(head)
            except:
                return False
def test():
    cln = CircularLinkedNode(start=2, modulo=2)
    for i in range(10):
        cln.add(i)
    assert cln.head.next.next.next.next.next.next.value == 2
    assert is_circular(cln) == True

    ll = LinkedList()
    for i in range(10):
        ll.add(i)
    assert is_circular(ll) == False
