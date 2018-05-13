'''
Python Implementation : Palindrome
'''

__author__ = "tchaton"

def test():
    sentence = 'hello there'
    assert(check_if_palindrome(sentence) == False)

    sentence = ''
    assert(check_if_palindrome(sentence) == True)

    sentence = 'h'
    assert(check_if_palindrome(sentence) == True)

    sentence = 'Noel sees Leon'
    assert(check_if_palindrome(sentence) == True)

    sentence = 'Noel sees Leon!'
    assert(check_if_palindrome(sentence) == True)

def clean(letter):
    remove = [' ', '!']
    if letter in remove:
        return False
    return True

def check_if_palindrome(sentence):
    seq = [letter.lower() for letter in list(sentence) if clean(letter)]
    n = len(seq) // 2
    for x, y in zip(seq[:n], seq[::-1][:n]):
        if x!=y:
            return False
    return True
