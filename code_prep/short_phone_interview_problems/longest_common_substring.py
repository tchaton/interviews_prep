'''
Python Implementation : Longest common substring
'''
__author__ = "tchaton"

def test():
    str1 = 'hasfgeaae'
    str2 = 'bafgekk'
    result = 'fge'
    assert(longest_common_substring(str1, str2) == result)

    str1 = 'bafgekk'
    str2 = 'hasfgeaae'
    result = 'fge'
    assert(longest_common_substring(str1, str2) == result)

    str1 = ''
    str2 = 'hasfgeaae'
    result = ''
    assert(longest_common_substring(str1, str2) == result)

def has_value(str, value):
    out = []
    for index, v in enumerate(str):
        if v == value:
            out.append(index)
    return out

def find_next(str1, str2):
    cnt = 0
    for x,y in zip(str1, str2):
        if x == y:
            cnt+=1
    return cnt

def find_substring(str1, str2):
    longest_substring = ''
    length_longest_substring = 0
    for index in range(len(str2)):
        token = str2[index]
        positions = has_value(str1, token)
        if len(positions) > 0:
            for pos in positions:
                cnt = find_next(str1[pos:], str2[index:])
                if length_longest_substring < cnt:
                    longest_substring = str1[pos:pos+cnt]
                    length_longest_substring = cnt
    return ''.join(longest_substring)

def longest_common_substring(string1, string2):
    str1 = list(string1)
    str2 = list(string2)
    if len(str1) == 0 or len(str2) == 0:
        return ''
    if len(str1) > len(str2):
        substring = find_substring(str1, str2)
    else:
        substring = find_substring(str2, str1)
    return substring
