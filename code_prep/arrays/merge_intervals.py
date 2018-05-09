"""
Given a collection of intervals, merge all overlapping intervals.
Everything is wrapped in a class, but it could be put out of it.
"""

import numpy as np
def test():

    intervals = [[1, 2], [5, 6], [1.5, 3], [5, 8], [5, 9], [5, 10]]
    out = [[1.0, 3], [5, 10]]
    assert Intersector(intervals).get_intervals == sorted(out, key=lambda x: x[0]), 'ERROR'

class Intersector:

    def __init__(self, intervals):
        self.intervals = intervals
        self.merge_intervals(self.intervals)

    @property
    def get_intervals(self):
        return self.intervals

    def check_end(self, sblb):
        for i in range(len(sblb) -1 ):
            if sblb[i][1] > sblb[i+1][0]:
                return False
        return True

    def delete_elements(self, l, le):
        out = []
        for i, e in enumerate(l):
            if i not in le:
                out.append(e)
        return out

    def merge_intervals(self, intervals):
        L = len(intervals)
        i, j = 0, 1
        while not self.check_end(sorted(intervals, key=lambda x: x[0])):
            intervals = sorted(intervals, key=lambda x: x[0])
            sblb = sorted(intervals, key=lambda x: x[0])
            a, b = sblb[i], sblb[j]
            bool_v , inter = self.intersect(a, b)
            if bool_v:
                intervals = self.delete_elements(intervals, [i, j])
                intervals.insert(0, inter)
            else:
                j = (j+1)
                i = (i+1)
                if j % L == 0:
                    i, j = 0, 1
        self.intervals = sorted(intervals, key=lambda x: x[0])

    def intersect(self, a, b):
        x, y = a
        w, z = b
        if not ((y <= w) or (x >= z)):
            return True, [np.min([x, w]), np.max([y, z])]
        else:
            return False, [a, b]
