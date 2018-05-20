class Node:

    def __init__(self, value, next=None):
        self.value = value
        self.next = next

    def _add(self, node):
        if self.next is not None:
            self.next._add(node)
        else:
            self.next = node

class LinkedList:

    def __init__(self):
        self.head = None

    def add(self, value):
        if self.head == None:
            self.head = Node(value)
        else:
            self.head._add(Node(value))
            
    def _printList(self):
        node = self.head
        while node.next != None:
            print(node.value)
            node = node.next
        print(node.value)
