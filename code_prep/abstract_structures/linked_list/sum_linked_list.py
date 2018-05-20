#!/usr/bin/env python

__author__ = "tchaton"


''' Supposing two linked lists representing numbers, such that in each of their
    nodes they carry one digit. This function sums the two numbers that these
    two linked lists represent, returning a third list representing the sum:'''

from .base import Node, LinkedList

def test():
    ll = LinkedList()
    List = [5, 6, 9, 98, 53, -1, 5, 6]
    for i in List:
        ll.add(i)

    ll2 = LinkedList()
    List = range(5)
    for i in List:
        ll2.add(i)

    l_sum = sum_linked_lists(ll, ll2)
    l_sum._printList()

    '''
    Result:
    7
    11
    101
    57
    -1
    5
    6
    '''

def iter(ll):
    node = ll.head
    while True:
        if node == None:
            yield 0, False
        else:
            node = node.next
            if node != None:
                yield node.value, True
            else:
                yield 0, False

def sum_linked_lists(ll, ll2):
    if ll.head == None:
        return ll2
    if ll2.head == None:
        return ll
    l_sum = LinkedList()
    for x, y in zip(iter(ll), iter(ll2)):
        v_ll, is_null = x
        v_ll2, is_null2 = y
        if l_sum.head == None:
            l_sum.head = Node(v_ll+v_ll2)
        else:
            if is_null or is_null2:
                l_sum.head._add(Node(v_ll+v_ll2))
            else:
                break
    return l_sum
