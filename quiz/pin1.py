# coding: utf-8

import sys


def sol():
    val = sys.stdin.readline().strip()[1:-1].split(',')
    # print(val)
    max_len = 0
    if len(val) == 1 and val[0] == '':
        print(0)
        return
    val = list(map(int, val))
    for i in range(1, len(val) - 1):
        a = val[i]
        r = a
        l = a
        bor = False
        for x in range(i - 1, -1, -1):
            if val[x] > r:
                r = val[x]
                bor = True
            else:
                bor = False
                break
        if bor:
            max_left = i - x
        else:
            max_left = i - x - 1
        for y in range(i + 1, len(val)):
            if val[y] > l:
                l = val[y]
                bor = True
            else:
                bor = False
                break
        if bor:
            max_right = y - i
        else:
            max_right = y - i - 1
        if max_left > 0 and max_right > 0:
            this_len = max_left + max_right + 1
        else:
            this_len = 0
        if this_len > max_len:
            max_len = this_len
    print(max_len)


if __name__ == "__main__":
    # 读取第一行的n
    sol()


