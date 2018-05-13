'''
Python code : Implementation Hash Table
'''

__author__ = "tchaton"


import hashlib

def test():
    dic = HashTable(5)
    for i in range(1, 40, 2):
        dic.add_item(i)

    dic.print_table()
    assert(dic.find_item(20) == False)
    assert(dic.find_item(21) == True)

class HashTable:

    def __init__(self, nb_bucket):
        if nb_bucket == None:
            self.nb_bucket = 100
        else:
            self.nb_bucket = nb_bucket
        self.holder = [[] for i in range(self.nb_bucket)]

    def hash_function(self, value):
        hash = hashlib.md5(str(value).encode()).hexdigest()
        hash_key = int(hash, 16) % self.nb_bucket
        return hash, hash_key

    def add_item(self, item):
        hash, hash_key = self.hash_function(item)
        self.holder[hash_key].append([hash, item])

    def print_table(self):
        print(self.holder)

    def find_item(self, item):
        hash, hash_key = self.hash_function(item)
        for ele in self.holder[hash_key]:
            if ele[0] == hash:
                return True
        return False
