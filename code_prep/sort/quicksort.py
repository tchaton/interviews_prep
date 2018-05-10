'''
Implementation quicksort
Complexity best : O(nlogn)
Complexity avg : O(nlogn)
Complexity worst : O(n^2)


'''

from ..utils import asssert_func

import numpy as np
def test():
    arr = np.random.randint(0, 100, (1000,))
    #sort = sorted(arr)
    #asssert_func(quicksort, arr, sort)

    assert quick_sort([1]) == [1]
    assert quick_sort([1,2]) == [1,2]
    assert quick_sort([1,2,3]) == [1,2,3]
    assert quick_sort([1,2,3,4]) == [1,2,3,4]
    assert quick_sort([2,1,3,4]) == [1,2,3,4]
    assert quick_sort([1,3,2,4]) == [1,2,3,4]
    assert quick_sort([1,2,4,3]) == [1,2,3,4]
    assert quick_sort([2,1,1,1]) == [1,1,1,2]
    assert quick_sort([1,2,1,1]) == [1,1,1,2]
    assert quick_sort([1,1,2,1]) == [1,1,1,2]
    assert quick_sort([1,1,1,2]) == [1,1,1,2]

# WITH EXTRA LISTS
def quicksort(arr):
    if len(arr) > 0:
        pivot = arr[0]

        below = [i for i in arr[1:] if i < pivot]
        right = [i for i in arr[1:] if i >= pivot]
        return quicksort(below) + [pivot] + quicksort(right)
    else:
        return arr

def quick_sort(a_list):
    """Hoare partition scheme, see https://en.wikipedia.org/wiki/Quicksort"""
    def _quicksort(a_list, low, high):
        # must run partition on sections with 2 elements or more
        if low < high:
            p = _partition(a_list, low, high)
            _quicksort(a_list, low, p)
            _quicksort(a_list, p+1, high)

    def _partition(a_list, low, high):
        pivot = a_list[low]
        while True:
            while a_list[low] < pivot:
                low += 1
            while a_list[high] > pivot:
                high -= 1
            if low >= high:
                return high
            a_list[low], a_list[high] = a_list[high], a_list[low]
            low += 1
            high -= 1

    _quicksort(a_list, 0, len(a_list)-1)
    return a_list

