"""
Given a string, find the length of the longest substring
without repeating characters.
Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.
Given "bbbbb", the answer is "b", with the length of 1.
Given "pwwkew", the answer is "wke", with the length of 3.
Note that the answer must be a substring,
"pwke" is a subsequence and not a substring.
"""

from ..utils import asssert_func

def insert_dict(d, key):
    try:
        d[key] += 1
    except:
        d[key] = 1
    return d[key]

def test():
    init = "abcabcbb"
    out = "abc"
    asssert_func(longest_non_repeat, init, out)

    init = "bbbbb"
    out = "b"
    asssert_func(longest_non_repeat, init, out)

    init = "pwwkew"
    out = "wke"
    asssert_func(longest_non_repeat, init, out)

def longest_non_repeat(string):
    string_list = list(string)
    letter_holder = []
    d = {}
    longest_size = 0
    longest_sub_e = ''
    for e in string_list:
        v = insert_dict(d, e)
        if v < 2:
            letter_holder.append(e)
        if v >= 2:
            if len(letter_holder) > longest_size:
                longest_sub_e = ''.join(letter_holder)
                longest_size = len(letter_holder)
                print(longest_sub_e)
            d = {}
            letter_holder = []
            v = insert_dict(d, e)
            letter_holder.append(e)
    return longest_sub_e
