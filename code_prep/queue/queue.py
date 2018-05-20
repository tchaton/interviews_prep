
#!/usr/bin/env python

__author__ = "bt3"

def test():
    queue = Queue()
    print("Is the queue empty? ", queue.isEmpty())
    print("Adding 0 to 10 in the queue...")
    for i in range(10):
        queue.enqueue(i)
    print("Queue size: ", queue.size())
    print("Queue peek : ", queue.peek())
    print("Dequeue...", queue.dequeue())
    print("Queue peek: ", queue.peek())
    print("Is the queue empty? ", queue.isEmpty())

    print("Printing the queue...")
    print(queue)

    '''
    Is the queue empty?  True
    Adding 0 to 10 in the queue...
    Queue size:  10
    Queue peek :  0
    Dequeue... 0
    Queue peek:  1
    Is the queue empty?  False
    Printing the queue...
    <code_prep.queue.queue.Queue object at 0x000002A05ABFFBE0>
    '''

class Queue(object):

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def isEmpty(self):
        return len(self.in_stack) == 0 and len(self.out_stack) == 0

    def enqueue(self, item):
        self.in_stack.append(item)

    def _transfer(self):
        while len(self.in_stack) >0:
            self.out_stack.append(self.in_stack.pop())

    def dequeue(self):
        if len(self.out_stack) == 0:
            self._transfer()
        if len(self.out_stack) > 0:
            return self.out_stack.pop()
        else:
            return "Queue empty!"

    def size(self):
        return len(self.in_stack) + len(self.out_stack)

    def peek(self):
        if len(self.out_stack) == 0:
            self._transfer()
        if len(self.out_stack) > 0:
            return self.out_stack[-1]
        else:
            return "Queue empty!"
