
#!/usr/bin/env python

__author__ = "tchaton"

""" A class for an animal shelter with two queues"""
from .queue import Queue

def test():
    qs = AnimalShelter()
    qs.enqueue('bob', 'cat')
    qs.enqueue('mia', 'cat')
    qs.enqueue('yoda', 'dog')
    qs.enqueue('wolf', 'dog')
    qs._print()

    print("Deque one dog and one cat...")
    qs.dequeue('dog')
    qs.dequeue('cat')
    qs._print()
    '''
    cat
    ['bob', 'mia']

    dog
    ['yoda', 'wolf']

    Deque one dog and one cat...
    cat
    ['mia']

    dog
    ['wolf']
    '''

class AnimalShelter:

    def __init__(self):
        self.shelter = {}

    def enqueue(self, name, kind):
        try:
            self.shelter[kind].enqueue(name)
        except:
            self.shelter[kind] = Queue()
            self.shelter[kind].enqueue(name)

    def _print(self):
        for key in self.shelter.keys():
            print(key)
            self.shelter[key]._print()
            print()

    def dequeue(self, kind):
         try:
             self.shelter[kind].dequeue()
         except:
            Exception('Error')
