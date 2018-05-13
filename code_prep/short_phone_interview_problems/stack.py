'''
Python Implementation : Stack
'''


__author__ = "tchaton"


def test():
    q = Stack()

    for i in range(10):
        q.push(i)
    for i in range(11):
        print(q.pop())

class Stack:

    def __init__(self):
        self.content = []

    def push(self, value):
        self.content.append(value)

    def pop(self):
        if len(self.content) > 0:
            value = self.content[-1]
            self.content = self.content[:-1]
            return value
