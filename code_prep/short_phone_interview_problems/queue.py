#!/usr/bin/env python

'''
Python Implementation : Queue
'''

__author__ = "tchaton"

def test():
    q = Queue()
    for i in range(1,11):
        q.enqueue(i)
    q.print_queue()
    print('Size:',  q.size())
    print('Is empty?', q.isempty())
    print('Peak: ', q.peak())
    q.print_queue()

class Queue:

    def __init__(self):
        self.content = []

    def enqueue(self, value):
        self.content.insert(0, value)

    def size(self):
        return len(self.content)

    def isempty(self):
        return len(self.content) == 0

    def print_queue(self):
        print(self.content)

    def peak(self):
        top = self.content[-1]
        self.content = self.content[:-1]
        return top
