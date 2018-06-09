# Count the number of ways to make change for a given number of cents.
__author__ = 'tchaton'

import unittest

def coins(n):
    count = 0
    for c in range(n, -1, -25):
        count+=coins_pnd(c)
    return count
def coins_pnd(c):
    count = 0
    for c in range(c, -1, -10):
        count += (c//5)+1
    return count

class Test(unittest.TestCase):
  def test_coins(self):
    self.assertEqual(coins(0), 1)
    self.assertEqual(coins(1), 1)
    self.assertEqual(coins(4), 1)
    self.assertEqual(coins(5), 2)
    self.assertEqual(coins(15), 6)
    self.assertEqual(coins(17), 6)
    self.assertEqual(coins(20), 9)
    self.assertEqual(coins(25), 13)
    self.assertEqual(coins(52), 49)


if __name__ == "__main__":
    unittest.main()
