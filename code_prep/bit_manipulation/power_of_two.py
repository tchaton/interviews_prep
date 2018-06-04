# Determine is power of two.

__author__ = 'tchaton'

def is_power_of_two_or_zero(n):
    return (n & (n -1)) == 0

import unittest

class Test(unittest.TestCase):
    def test_is_power_of_two(self):
        self.assertEqual(is_power_of_two_or_zero(1024), True)
        self.assertEqual(is_power_of_two_or_zero(655), False)

if __name__ == "__main__":
    unittest.main()
