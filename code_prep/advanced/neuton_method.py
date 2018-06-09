# Implementation of Neuton Derivative

__author__ = 'tchaton'

import unittest
import numpy as np

def decay(cnt):
    if cnt == 0:
        return 1
    else:
        return 1#/cnt

def neuton_method(f, f2, start, error=10e-15):
    '''
    Reference : https://en.wikipedia.org/wiki/Newton%27s_method
    '''
    x_n = start
    cnt = 0
    while True:
        if np.abs(f(x_n)) < error:
            return x_n, f(x_n)
        else:
            x_n = x_n - decay(cnt)*(f(x_n)/f2(x_n))
            cnt+=1

class Test_Neuton(unittest.TestCase):

    def test_neuton(self):
        def f(x):
            return (x**3) - 2*x - 5
        def f2(x):
            return 3*(x**2) - 2
        self.assertEqual(neuton_method(f, f2, 10), (2.0945514815423265, -8.881784197001252e-16))

if __name__ == '__main__':
    unittest.main()
