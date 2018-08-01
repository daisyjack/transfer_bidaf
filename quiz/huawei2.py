# coding: utf-8

import sys

def process_neg(exp):
    idx = []
    for i, a in enumerate(exp):
        if a == '-':
            idx.append(i)
    if len(idx) == 1:
        a, b = exp.split('-')
        result = float(a) - float(b)
    elif len(idx) == 2:
        if idx[0] == 0:
            a = float(exp[0:idx[1]])
            b = float(exp[idx[1]+1:])
            result = a - b
        else:
            a = float(exp[0:idx[0]])
            b = float(exp[idx[1]:])
            result = a - b
    else:
        a = float(exp[0:idx[1]])
        b = float(exp[idx[2]:])
        result = a - b
    return result



if __name__ == "__main__":
    # 读取第一行的n
    exp = sys.stdin.readline().strip()
    if '+' in exp:
        a, b = exp.split('+')
        result = float(a) + float(b)
    elif '*' in exp:
        a, b = exp.split('*')
        result = float(a) * float(b)
    elif '/' in exp:
        a, b = exp.split('/')
        result = float(a) / float(b)
    elif '-' in exp:
        result = process_neg(exp)
    print(int(result))
    # print(eval(exp))


