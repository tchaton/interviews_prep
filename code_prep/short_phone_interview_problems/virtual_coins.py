# Count the number of ways to make change for a given number of cents.
__author__ = 'tchaton'

'''
Reference : https://www.geeksforgeeks.org/dynamic-programming-set-7-coin-change/
C() --> count()
                             C({1,2,3}, 5)
                           /             \
                         /                 \
             C({1,2,3}, 2)                 C({1,2}, 5)
            /       \                      /      \
           /         \                    /         \
C({1,2,3}, -1)  C({1,2}, 2)        C({1,2}, 3)    C({1}, 5)
               /    \             /     \           /     \
             /       \           /       \         /        \
    C({1,2},0)  C({1},2)   C({1,2},1) C({1},3)    C({1}, 4)  C({}, 5)
                   / \     / \        /\         /     \
                  /   \   /   \     /   \       /       \
                .      .  .     .   .     .   C({1}, 3) C({}, 4)
                                               / \
                                              /   \
                                             .      .
'''
import unittest

def count(S, m, n ):
    if n == 0:
        return 1
    if (n < 0):
        return 0
    if (m <=0 and n >=1 ):
        return 0
    return count(S, m - 1, n) + count(S, m, n - S[m-1])

class Test(unittest.TestCase):
  def test_coins(self):
    arr = [1, 2, 3]
    m = len(arr)
    print(count(arr, m, 4))


if __name__ == "__main__":
    unittest.main()
