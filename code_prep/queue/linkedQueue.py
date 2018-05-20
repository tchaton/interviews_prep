
#!/usr/bin/env python

__author__ = "tchaton"

''' Queue acts as a container for nodes (objects) that are inserted and removed according FIFO'''

from .queue import Queue


def test():
    queue = LinkedQueue()
    print("Is the queue empty? ", queue.isEmpty())
    print("Adding 0 to 10 in the queue...")
    for i in range(10):
        queue.enqueue(i)
    print("Is the queue empty? ", queue.isEmpty())
    queue._print_next()
    print()
    queue._print_previous()

    print("Queue size: ", queue.size())
    print("Queue peek : ", queue.peek())
    print("Dequeue...", queue.dequeue())
    print("Queue peek: ", queue.peek())
    queue._print_next()
    
    '''
    Is the queue empty?  True
    Adding 0 to 10 in the queue...
    Is the queue empty?  False
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9

    9
    8
    7
    6
    5
    4
    3
    2
    1
    0
    Queue size:  10
    Queue peek :  9
    Dequeue... 9
    Queue peek:  8
    0
    1
    2
    3
    4
    5
    6
    7
    8
    '''


class Node:

    def __init__(self, value=None, previous=None, next=None):
        self.value = value
        self.previous = previous
        self.next = next

class LinkedQueue(object):

    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return self.head == None and self.tail == None

    def enqueue(self, value):
        if not self.head:
            node = Node(value=value)
            self.head = node
            self.tail = node
        else:
            node =  Node(value=value, previous=self.tail)
            if self.tail:
                self.tail.next = node
            self.tail = node

    def size(self):
        cnt = 0
        node = self.head
        while node:
            cnt+=1
            node = node.next
        return cnt

    def _print_next(self):
        node = self.head
        while node:
            print(node.value)
            node = node.next

    def _print_previous(self):
        node = self.tail
        while node:
            print(node.value)
            node = node.previous

    def peek(self):
        if self.tail:
            return self.tail.value

    def dequeue(self):
        if self.tail:
            tail = self.tail
            previous = tail.previous
            previous.next = None
            self.tail = previous
            return tail.value
        else:
            Exception('Can t dequeue, empty list')
