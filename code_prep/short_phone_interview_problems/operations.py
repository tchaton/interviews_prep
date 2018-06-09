# Implement operations with only addition (and negation).

def multiply(a, b):
    if b == 0:
        return 0
    return a + multiply(a, b - 1)

def divide(a, b):
    product = 0
    quotient = 0
    while product + abs(b) < abs(a) + 1:
        product+=abs(b)
        quotient+=1
    if a < 0 or b < 0:
        return -quotient
    return quotient

def subtract(a, b):
    return a + (-b)

import unittest

class Test(unittest.TestCase):
  def test_multiply(self):
    self.assertEqual(multiply(3, 6), 18)
    self.assertEqual(multiply(7, 11), 77)
    self.assertEqual(multiply(-8, 10), -80)

  def test_subtract(self):
    self.assertEqual(subtract(34, 6), 28)
    self.assertEqual(subtract(31, -6), 37)

  def test_divide(self):
    self.assertEqual(divide(3, 6), 0)
    self.assertEqual(divide(30, 6), 5)
    self.assertEqual(divide(34, -6), -5)
    self.assertEqual(divide(120, 10), 12)


if __name__ == "__main__":
    unittest.main()
