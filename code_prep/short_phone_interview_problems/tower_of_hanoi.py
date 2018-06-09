# Move the blocks on tower1 to tower3.

class Tower(object):
    def __init__(self, name, discs=[]):
        self.name = name
        self.discs = discs

    def pop(self):
        return self.discs.pop()

    def append(self, item):
        self.discs.append(item)

class Towers_of_hanoi(object):

    def __init__(self, towers=[], source='Tower1', target='Tower3', auxiliary='Tower2'):
        self.towers = {tower.name:tower for tower in towers}
        self.source = source
        self.target = target
        self.auxiliary = auxiliary

    def solve(self):
        def movestack(disk, source, dest, temp):
            if disk == 1:
                dest.append(source.pop())
            else:
                movestack(disk - 1, source, temp, dest)
                dest.append(source.pop())
                movestack(disk - 1, temp, dest, source)

        n = len(self.towers[self.source].discs)
        print(n)
        target = self.towers[self.target].discs
        source = self.towers[self.source].discs
        auxiliary = self.towers[self.auxiliary].discs

        movestack(n, source, target, auxiliary)
        self.towers[self.auxiliary].discs = []
        self.towers[self.target].discs = self.towers[self.target].discs[::-1]

    def _print(self, n):
        print(n)
        for name in self.towers.keys():
            print(name, self.towers[name].discs)

import unittest

class Test(unittest.TestCase):
    def test_towers_of_hanoi(self):
        tower1 = Tower("Tower1", ["6", "5", "4", "3", "2", "1"])
        tower2 = Tower("Tower2")
        tower3 = Tower("Tower3")
        towers = Towers_of_hanoi([tower1, tower2, tower3])
        towers.solve()
        self.assertEqual(tower1.discs, [])
        self.assertEqual(tower2.discs, [])
        self.assertEqual(tower3.discs, ["6", "5", "4", "3", "2", "1"])

if __name__ == "__main__":
    unittest.main()
