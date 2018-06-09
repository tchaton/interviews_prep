
# Find the year with the most living people.
import numpy as np

def most_living_people(people):
    np.min([p.birth_year for p in people])
    np.max([p.death_year for p in people])

class P(object):
    def __init__(self, born, died=None):
        self.birth_year = born
        self.death_year = died

import unittest
class Population:
    def __init__(self, N=100):
        self.N = N
        self.create_population()

    def create_population(self, start_year=1900, end_year=2018, life_expectancy=90):
        pop = []
        for _ in range(self.N):
            birthday = np.random.randint(start_year, end_year)
            life_time = np.random.randint(1, life_expectancy)
            pop.append(P(birthday, birthday + life_time))
        self.pop = pop

    def find_max(self):
        dates = sorted([(p.birth_year, 0) for p in self.pop]+[(p.death_year, 1) for p in self.pop], key=lambda x:x[0])
        cnt = 0
        _max = 0
        max_date = None
        intervals = []
        for d in dates:
            if d[1] == 0:
                cnt +=1
            elif d[1] == 1:
                cnt -=1
            if cnt > _max:
                intervals.append([max_date, d[0], cnt])
                _max = cnt
                max_date = d[0]
        print('Intervals :',intervals)
        print('MAX AND DATE:', _max, max_date)

class Test(unittest.TestCase):
    def test_most_living_people(self):
        pop = Population()
        pop.find_max()
        #self.assertEqual(most_living_people(people), 1933)

if __name__ == "__main__":
    unittest.main()
