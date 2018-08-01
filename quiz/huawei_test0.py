# coding: utf-8

import sys

def is_2_joker(card_lst):
    if 'joker' in card_lst and 'JOKER' in card_lst:
        return True
    else:
        return False


def compare(a_lst, b_lst):
    card_map = {'3':3, '4':4, '5':5, '6':6, '7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13,'A':14,'2':15,'joker':16,'JOKER':17}
    card_type = ['single', 'double', 'seq', 'bomb', '2_joker']
    if is_2_joker(a_lst):
        return 'a'
    if is_2_joker(b_lst):
        return 'b'
    if len(a_lst) != len(b_lst):
        if len(a_lst) == 4:
            return 'a'
        elif len(b_lst) == 4:
            return 'b'
        else:
            return 'no'
    else:
        a = card_map[a_lst[0]]
        b = card_map[b_lst[0]]
        if a>b:
            return 'a'
        else:
            return 'b'






if __name__ == "__main__":
    # 读取第一行的n
    cal = sys.stdin.readline().strip()
    # cal = 'a-b'
    a, b = cal.split('-')
    a_lst = a.split(' ')
    b_lst = b.split(' ')
    result = compare(a_lst, b_lst)
    if result == 'no':
        print('ERROR')
    elif result == 'a':
        print(a)
    elif result == 'b':
        print(b)




