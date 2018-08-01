# coding: utf-8

import sys



if __name__ == "__main__":
    # 读取第一行的n
    val = sys.stdin.readline().strip()
    min_l = 100
    for i in range(0, len(val)):
        can = val[0:i+1]
        p = int(len(val) / len(can))
        mod = len(val) - len(can)*p
        can_a = can*p + can[0:mod]
        if can_a == val:
            this_min = len(can)
            if this_min < min_l:
                min_l = this_min
                break
    print(can)

