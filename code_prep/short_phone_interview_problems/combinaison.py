'''
Python Code for Combinaison
'''

__author__ = "tchaton"

def test():
    array = ['a', 'b', 'c']
    result = set(['', 'c', 'b', 'bc', 'a', 'ac', 'ab', 'abc'])
    assert(combination(array) == result)

def fact(n):
    """fact(n): calcule la factorielle de n (entier >= 0)"""
    if n<2:
        return 1
    else:
        return n*fact(n-1)

def comb(n, k):
    return fact(n)/(fact(k)*fact(n-k))

def int2bin(val):
   res=''
   while val>0:
       res = chr((val&1) + 0x30) + res
       val=val>>1
   if len(res) == 1:
       return '00'+res
   if len(res) == 2:
       return '0'+res
   else:
       return res
def create_masks(array):
    L = len(array)
    nb_combs = int(sum([comb(L,i) for i in range(0, L+1)]))
    return [int2bin(i) for i in range(1, nb_combs)]

def combination(array):
    L = len(array)
    if L < 2:
        return array
    masks = create_masks(array)
    comb = ['']
    for mask in masks:
        mask_list = list(mask)
        e = ''.join([x for x, i in zip(array, mask_list) if int(i) ==1])
        comb.append(e)
    return set(comb)
