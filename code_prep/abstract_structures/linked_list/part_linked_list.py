#!/usr/bin/env python

__author__ = "tchaton"

''' This function divides a linked list in a value, where everything smaller than this value
    goes to the front, and everything large goes to the back:'''

from .base import Node, LinkedList

def test():
    ll = LinkedList()
    List = [5, 6, 9, 98, 53, -1, 5, 6]
    for i in List:
        ll.add(i)
    lleft, lright =  partList(ll, 10)
    #return lleft, lright
    lleft._printList()
    print()
    lright._printList()
    '''
    Result :
    5
    6
    9
    -1
    5

    98
    53
    '''

def partList(ll, n):
    if ll.head == None:
        return None
    else:
        node = ll.head
        lleft = LinkedList()
        lright = LinkedList()

        while node.next != None:
            value = node.value

            if value < n:
                if lleft.head == None:
                    lleft.head = Node(value)
                else:
                    lleft.head._add(Node(value))
            else:
                if lright.head == None:
                    lright.head = Node(value)
                else:
                    lright.head._add(Node(value))
            node = node.next
        return lleft, lright
