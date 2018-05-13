'''
Python code Implementation: Permutation
'''


__author__ = "tchaton"

def test():
    word = 'abc'
    result = ['abc', 'acb', 'bac', 'bca', 'cab', 'cba']
    assert(permutation(word) == result)

    word = ''
    result = ['']
    assert(permutation(word) == result)

    word = 'a'
    result = ['a']
    assert(permutation(word) == result)

def permutation(str, depth = 0):
    if len(str) < 2:
        return [str]

    result = []
    for index, letter in enumerate(str):
        new_str = str[:index] + str[index+1:]
        for perm in permutation(new_str, depth=depth+1):
            result.append(letter+perm)
    return result
