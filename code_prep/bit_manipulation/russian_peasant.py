# Multiply two positive integers without *.

__author__ = 'tchaton'

# O(lg(min(a, b)))

def multiply(a, b):
    '''
    Russian peasant multiplication
    Reference : https://en.wikipedia.org/wiki/Ancient_Egyptian_multiplication#Russian_peasant_multiplication
    '''

    if b > a: a, b = b, a
    res = 0 # initialize result

    while (b > 0):

        if (b & 1):
            res = res + a

        a = a << 1
        b = b >> 1

    return res

import unittest

class Test(unittest.TestCase):
    def test_multiply(self):
        self.assertEqual(multiply(2, 2), 4)
        self.assertEqual(multiply(1, 125), 125)
        self.assertEqual(multiply(7, 11), 77)
        self.assertEqual(multiply(10000000010, 21), 210000000210)

if __name__ == "__main__":
    unittest.main()
