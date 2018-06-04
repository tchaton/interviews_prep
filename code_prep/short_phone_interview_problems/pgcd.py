#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Reference : https://en.wikipedia.org/wiki/Euclidean_algorithm
http://python.jpvweb.com/python/mesrecettespython/doku.php?id=pgcd_ppcm
'''

__author__ = 'tchaton'
import numpy as np

def pgcd(a, b):
    a = np.abs(a)
    b = np.abs(b)

    while b != 0:
        a, b = b, a%b
    return a

def pgcdn(*n):
    p = pgcd(n[0], n[1])
    for x in n[2:]:
        p = pgcd(p, x)
    return p

def pgcde(a, b):
    r, u, v = a, 1, 0
    rp, up, vp = b, 0, 1
    while rp != 0:
        q = r//rp
        rs, us, vs = r, u, v
        r, u, v = rp, up, vp
        rp, up, vp = (rs - q*rp), (us - q*up), (vs - q*vp)
    return (r, u, v)

def ppcm(a, b):
    if (a == 0) or (b == 0):
        return 0
    else:
        return (a*b)//pgcd(a, b)


import unittest

class Test(unittest.TestCase):

    def test_pgcd(self):
        self.assertEqual(pgcd(56, 42), 14)
        self.assertEqual(pgcd(56, 0), 56)
        self.assertEqual(pgcd(11, 3), 1)
        self.assertEqual(pgcd(10, 5), 5)
        self.assertEqual(pgcd(542284229916, 231414210846), 1038258)

    def test_pgcdn(self):
        a = 30   # 2*3*5
        b = 462  # 2*3*7*11
        c = 390  # 2*3*5*13
        d = 798  # 2*3*7*19
        self.assertEqual(pgcdn(a, b, c, d), 6)

    def test_pgcde(self):
        self.assertEqual(pgcde(11, 3), (1, -1, 4))
        self.assertEqual(pgcde(56, 0), (56, 1, 0))
        self.assertEqual(pgcde(10, 5), (5, 0, 1))

    def test_ppcm(self):
        self.assertEqual(ppcm(56,0), 0)
        self.assertEqual(ppcm(56,42), 168)

if __name__ == '__main__':
    unittest.main()
