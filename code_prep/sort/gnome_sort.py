#!/usr/bin/env python

__author__ = "tchaton"


def test():
    seq = [3, 5, 2, 6, 8, 1, 0, 3, 5, 6, 2, 5, 4, 1, 5, 3]
    assert(gnome_sort(seq) == sorted(seq))

def gnome_sort(a):
    pos = 0
    L = len(a)
    while pos < L:
        if (pos == 0) or (a[pos] >= a[pos -1]):
            pos+=1
        else:
            a[pos], a[pos-1] = a[pos-1], a[pos]
            pos-=1
    return a
