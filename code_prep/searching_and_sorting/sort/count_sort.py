#!/usr/bin/env python

__author__ = "tchaton"


def test():
    seq = [3, 5, 2, 6, 8, 1, 0, 3, 5, 6, 2, 5, 4, 1, 5, 3]
    assert(count_sort_dict(seq) == sorted(seq))

def manage_dict(d, key):
    try:
        d[key]+=1
    except:
        d[key]= 1

def count_sort_dict(a):
    ''' an example of counting sort using default dictionaries '''
    d = {}
    sorted_arr = []
    for x in a:
        manage_dict(d, str(x))
    while len(d.keys()) > 0:
        keys = [int(key) for key in d.keys()]
        key_min = min(keys)
        cnt = d[str(key_min)]
        for _ in range(cnt):
            sorted_arr.append(key_min)
        del d[str(key_min)]
    return sorted_arr
