"""
Implement Flatten Arrays.
Given an array that may contain nested arrays,
produce a single resultant array.
"""

from collections import Iterable


# return list
def flatten(input_arr, output_arr=None):
    if isinstance(output_arr, type(None)):
        output_arr = []
    for e in input_arr:
        if isinstance(e, list):
            flatten(e, output_arr)
        else:
            output_arr.append(e)
    return output_arr

def test():
    import numpy as np
    arr = np.random.normal(0, 1, (5, 5, 5))
    prod = np.product(arr.shape)
    flatten_prod = np.product(np.array(flatten(arr.tolist())).shape)
    assert prod == flatten_prod, 'Error'
