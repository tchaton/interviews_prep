"""
Balanced string with parentheses
"""

from ..utils import asssert_func

def inverse_string(string):
    return ''.join(list(string)[::-1])

def test():
    
    asssert_func(balance, ')((', '()(())')
    asssert_func(balance, '))(', '(())()')
    asssert_func(balance, ')((((()(', '()((((()))))()')
    asssert_func(balance, inverse_string(')((((()('), '()((((()))))')
    asssert_func(balance, ')', '()')
    asssert_func(balance, '', '')

open = '('
close = ')'

def check_number(strings_list):
    cnt_open = 0
    cnt_close = 0
    for letter in strings_list:
        if letter == open:cnt_open +=1
        else:cnt_close+=1
    return cnt_open, cnt_close


def balanced(strings_list, cnt_open_total, cnt_close_total, reverse=False):
    cnt_open = 0
    cnt_close = 0
    out_strings = []
    inserted_indexes = []
    inserted = 0
    diff = cnt_open_total - cnt_close_total
    if not reverse:
        for index, letter in enumerate(strings_list):
            if letter == open:
                out_strings.append(open)
                cnt_open+=1
                if strings_list[index + 1] == close:
                    cnt = cnt_open - cnt_close - 1
                    for _ in range(cnt):
                        out_strings.append(close)
                        cnt_close+=1
            else:
                out_strings.append(close)
                cnt_close+=1
    else:
        strings_list_inverted = strings_list[::-1]
        for index, letter in enumerate(strings_list_inverted):
            if letter == close:
                out_strings.insert(0, close)
                cnt_close+=1
                if strings_list_inverted[index + 1] == open:
                    cnt = cnt_close - cnt_open - 1
                    for _ in range(cnt):
                        out_strings.insert(0, open)
                        cnt_open+=1
            else:
                out_strings.insert(0, open)
                cnt_open+=1
    return out_strings

def check_start_end(strings_list):
    end_with_open = False
    start_with_close = False
    if strings_list[0] == close:
        start_with_close = True
    if strings_list[-1] == open:
        end_with_open = True
    return start_with_close, end_with_open

def balance(string):
    if string == None or string=='':
        return ''
    strings_list = list(string)
    if len(strings_list) == 0:
        return ''
    start_with_close, end_with_open = check_start_end(strings_list)
    if start_with_close:
        strings_list.insert(0, open)
    if end_with_open:
        strings_list.append(close)
    cnt_open_total, cnt_close_total = check_number(strings_list)
    if cnt_open_total >= cnt_close_total:
        out_strings = balanced(strings_list, cnt_open_total, cnt_close_total, reverse=False)
    else:
        out_strings = balanced(strings_list, cnt_open_total, cnt_close_total, reverse=True)
    output = ''.join(out_strings)
    print(string+' -> '+output)
    return output
