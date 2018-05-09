"""
Given a collection of intervals, merge all overlapping intervals.
"""

from ..utils import asssert_func

def test():

    intervals = [[1, 2], [1.5, 3], [5, 6]]
    out = [[1, 3], [5, 6]]
    asssert_func(merge_intervals, intervals, out)

def merge_intervals(intervals):
    pass
