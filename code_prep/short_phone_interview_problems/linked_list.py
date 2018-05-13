'''
Python code Implementing a Linded List
'''

__author__ = "tchaton"

def test():
    ll = LinkedList()
    for i in range(0, 5):
        ll.add(i)

    print('The list is:')
    ll._printList()

    print('The list after deleting node with index 2:')
    ll.deleteNode(2)
    ll._printList()

class Node(object):

    def __init__(self, value, next=None):
        self.value = value
        self.next = next

    def add(self, node):
        if self.next == None:
            self.next = node
        else:
            self.next.add(node)

class LinkedList:

    def __init__(self):
        self.head = None

    def add(self, value):
        if self.head == None:
            self.head = Node(value)
        else:
            self.head.add(Node(value))

    def _printList(self):
        node = self.head
        while node.next != None:
            print(node.value)
            node = node.next
        print(node.value)

    def deleteNode(self, index):
        cnt = 1
        node = self.head
        node_prev = None
        try:
            for _ in range(index):
                node_prev = node
                node = node.next
            node_prev.next = node.next
        except:
            print('This index can t be found')
