'''
Python Code to check for anagrams. Double list
'''

def test():
    word1 = 'abc'
    word2 = 'bca'
    assert(check_if_anagram(word1, word2) == True)

    word2 = 'bcd'
    assert(check_if_anagram(word1, word2) == False)

    word1 = ''
    word2 = ''
    assert(check_if_anagram(word1, word2) == True)

    word1 = 'a'
    word2 = 'a'
    assert(check_if_anagram(word1, word2) == True)

def fillup(key, d):
    try:
        d[key]+=1
    except:
        d[key] = 1

def check_if_anagram(word1, word2):
    word1 = list(word1)
    word2 = list(word2)
    if len(word1) != len(word2):
        return False
    d = {}
    d2 = {}
    for x, y in zip(word1, word2):
        fillup(x, d)
        fillup(y, d2)
    cnd = set(d.items()) ^ set(d2.items())
    return len(cnd) == 0
