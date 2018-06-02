
__author__ = 'tchaton'

import unittest

def rotated_search(arr, item):
    f = arr[0]
    seen = []
    def _rotated_search(arr, item, s, e, order=True):
        if s == e:
            return False
        mid = (s + e)//2
        item_mid = arr[mid]
        if item_mid in seen:
            return False
        seen.append(item_mid)
        if item_mid < item:
            return _rotated_search(arr, item, mid, e)
        elif item_mid > item:
            if item_mid < f and item < f:
                return _rotated_search(arr, item, s, mid)
            elif item_mid > f and item < f:
                return _rotated_search(arr, item, mid, e)
            else:
                return _rotated_search(arr, item, s, mid)
        else:
            return True
    return _rotated_search(arr, item, 0, len(arr) - 1)

class Test(unittest.TestCase):

    def test_sparse_test(self):
        # Search for an item in a rotated array.
        array = [55, 60, 65, 70, 75, 80, 85, 90, 95, 15, 20, 25, 30, 35, 40, 45]
        #self.assertEqual(rotated_search(array, 55), True)
        #self.assertEqual(rotated_search(array, 20), True)
        #self.assertEqual(rotated_search(array, 35), True)
        #self.assertEqual(rotated_search(array, 60), True)
        #self.assertEqual(rotated_search(array, 70), True)
        self.assertEqual(rotated_search(array, -1), False)

if __name__ == '__main__':
    unittest.main()
