"""
There are people sitting in a circular fashion,
print every third member while removing them,
the next counter starts immediately after the member is removed.
Print till all the members are exhausted.
For example:
Input: consider 123456789 members sitting in a circular fashion,
Output: 369485271
"""

def test():
    init = 123456789
    out = 369485271
    assert josephus(init) == out, 'Error'

def josephus(init):
    init = list(str(init))
    out = []
    cnt = 0
    while len(init) != 0:
        cnt+=3
        cnt=(cnt-1)%len(init)
        out.append(init[cnt])
        del init[cnt]
    return int(''.join(out+init))
