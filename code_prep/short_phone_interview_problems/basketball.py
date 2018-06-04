#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'tchaton'
import numpy as np
import numpy as np

def simulate(step, N, shot=3):
    arr = []
    for i in range(N):
        h = []
        for _ in range(shot):
            if np.random.uniform(0, 1) < step:
                h.append(1)
            else:
                h.append(0)
        arr.append(np.sum(h)/len(h))
    return np.mean(arr)

def estimate_proba_shot(s=0, e=1, N=1000):
    steps = np.linspace(0, 1, 100)
    probas = []
    for step in steps:
        proba = simulate(step, N)
        probas.append(proba)
    index = np.where(probas > steps)[0][0]
    return probas[index]

import unittest

class Test(unittest.TestCase):

    def test_shot(self):
        self.assertLess(estimate_proba_shot(), 0.035)

if __name__ == '__main__':
    unittest.main()
