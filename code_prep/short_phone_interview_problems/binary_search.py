'''
Binary Search
'''

from ..utils import asssert_func

def test():

    array = [3, 4, 6, 7, 10, 11, 34, 67, 84]
    value = 6
    assert(binary_search_rec(array, value) == True)
    value = 8
    assert(binary_search_rec(array, value) == False)
    array = [8]
    assert(binary_search_rec(array, value) == True)
    array = []
    assert(binary_search_rec(array, value) == False)

def binary_search_rec(arr, value):
    if len(arr) == 0:
        return False
    n = len(arr) // 2
    mid = arr[n]
    if mid < value:
        return binary_search_rec(arr[n+1:], value)
    elif mid == value:
        return True
    else:
        return binary_search_rec(arr[:n-1], value)
