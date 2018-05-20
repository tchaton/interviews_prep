#!/usr/bin/env python

__author__ = "tchaton"

from ..heap.heap import Heap
import numpy as np

def test():
    seq1 = [1, 2, 3, 8, 9, 10]
    seq2 = [2, 3, 4, 5, 6, 7, 9]
    seq3 = seq1 + seq2
    assert(merge_sorted_seq(seq1, seq2) == sorted(seq3))
    '''
    10
    9
    9
    8
    7
    6
    5
    4
    3
    3
    2
    2
    1
    '''

def generator(seq1, seq2):
    seq1 = seq1 [::-1]
    seq2 = seq2[::-1]
    cnt_seq1 = 0
    cnt_seq2 = 0
    L1 = len(seq1) - 1
    L2 = len(seq2) - 1
    while cnt_seq1 <= L1 and  cnt_seq2 <= L2 :
        if seq1[cnt_seq1] > seq2[cnt_seq2]:
            cnt_seq1+=1
            yield seq1[cnt_seq1 - 1]
        else:
            cnt_seq2+=1
            yield seq2[cnt_seq2 - 1]
    if cnt_seq1 == L1:
        for x in  seq2[cnt_seq2:]:
            yield x
    else:
        for x in  seq1[cnt_seq1:]:
            yield x

def merge_sorted_seq(seq1, seq2):
    '''
    Merge two sorted lists while trying to reduce the number of operations:
    Idea: Send back the biggest remaining value between two lists in the style of merge sort
    It will reduce the number of operations while inserting a value in the heap.
    '''
    heap = Heap()
    for x in generator(seq1, seq2):
        print(x)
        heap.add(x)
    return heap.heap[::-1]
