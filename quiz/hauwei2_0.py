# coding: utf-8

import sys



if __name__ == "__main__":
    # 读取第一行的n
    exp = sys.stdin.readline().strip()
    nums = [str(i) for i in range(10)]
    has_num = False
    for i, a in enumerate(exp):
        if a in nums:
            has_num = True
            continue
        elif has_num:
            break
    a = float(exp[0:i])
    b = float(exp[i+1:])
    op = exp[i]
    if op == '+':
        print(int(a+b))
    elif op == '-':
        print(int(a-b))
    elif op == '*':
        print(int(a*b))
    elif op == '/':
        print(a/b)


