
__author__ = 'tchaton'

import unittest

def sparse_search(arr, item):
    seen = []
    def _sparse_search(arr, item, s, e, order=True):
        print(s, e)
        if s == e:
            return False
        mid = (s + e)//2
        item_mid = arr[mid]
        while item_mid == 0:
            if order:
                mid +=1
                item_mid = arr[mid]
            else:
                mid -=1
                item_mid = arr[mid]
        if item_mid in seen:
            return False
        seen.append(item_mid)
        if item_mid < item:
            print('more')
            return _sparse_search(arr, item, mid, e)
        elif item_mid > item:
            print('less')
            return _sparse_search(arr, item, s, mid-1, order=False)
        else:
            return True
    return _sparse_search(arr, item, 0, len(arr) - 1)

class Test(unittest.TestCase):

    def test_sparse_test(self):
        # Search a sorted sparse array of real.
        array = [-1, 0, 0, 7, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 37, 40, 0, 0, 0]
        print(sparse_search(array, 7))
        self.assertEqual(sparse_search(array, 7), True)
        self.assertEqual(sparse_search(array, 37), True)
        self.assertEqual(sparse_search(array, -1), True)
        self.assertEqual(sparse_search(array, 10), False)
        self.assertEqual(sparse_search(array, -2), False)

if __name__ == '__main__':
    unittest.main()
