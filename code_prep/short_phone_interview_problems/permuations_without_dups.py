# List all permutations of a string that contains no duplicate letters.
def permutation(str):
    def partial_permutations(str):
        if len(str) < 2:
            return [str]

        result = []
        for index, letter in enumerate(str):
            new_str = str[:index] + str[index+1:]
            for perm in partial_permutations(new_str):
                result.append(letter+perm)
        return result
    return list(set(partial_permutations(str)))

import unittest

class Test(unittest.TestCase):
    def test_permutations(self):
        self.assertEqual(permutation("ABCD"), list(set(["ABCD", "ABDC", "ACBD", "ACDB",
            "ADBC", "ADCB", "BACD", "BADC", "BCAD", "BCDA", "BDAC", "BDCA",
            "CABD", "CADB", "CBAD", "CBDA", "CDAB", "CDBA", "DABC", "DACB",
            "DBAC", "DBCA", "DCAB", "DCBA"])))
        self.assertEqual(permutation("abca"), list(set(["aabc", "aacb", "abac", "abca",
    "acab", "acba", "baac", "baca", "bcaa", "caab", "caba", "cbaa"])))

if __name__ == "__main__":
    unittest.main()
