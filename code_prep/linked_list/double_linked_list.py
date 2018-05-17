__author__ = "tchaton"

''' Implement a double-linked list, which is very simple, we just need inherits
from a Linked List Class and  add an attribute for previous.'''

from .base import LinkedList

def test():
    dll = DoubleLinkedList()
    for i in range(10):
        dll.add(i)
    assert dll.head.next.next.previous.previous == dll.head
    dll._printList()
    dll.delete_node(0)
    print()
    dll._printList()
    dll.delete_node(5)
    print()
    dll._printList()

class dNode:

    def __init__(self, value=None, previous=None, next=None):
        self.value = value
        self.previous = previous
        self.next = next

    def _add(self, value):
        if self.next == None:
            self.next = dNode(value=value, previous=self)
        else:
            self.next._add(value)
        return self.next

class DoubleLinkedList(LinkedList):

    def __init__(self):
        self.tail = None
        super(DoubleLinkedList, self).__init__()

    def add(self, value):
        if self.head == None:
            self.head = dNode(value=value, previous=None)
            self.tail = self.head
        else:
            self.tail = self.head._add(value)

    def _printList(self):
        node = self.head
        while node.next != None:
            print(node.value)
            node = node.next
        print(node.value)

    def delete_node(self, index):
        if self.head == None:
            if index > 0:
                raise Exception('Trying to delete an un-existing node')
        node = self.head
        for _ in range(1, index):
            node = node.next

        if node.next == None:
            previous = node.previous
            previous.next = None
            self.tail = previous

        elif node.previous == None:
            next = node.next
            next.previous = None
            self.head = next
            
        else:
            previous = node.previous
            next = node.next
            next.previous = previous
            previous.next = next
