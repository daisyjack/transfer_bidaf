# coding: utf-8

import sys



if __name__ == "__main__":
    # 读取第一行的n
    val = sys.stdin.readline().strip().split(' ')
    y_val = []
    for a in val:
        if 'S' in a:
            money = int(a[0:-1])*7
        else:
            money = int(a[0:-1])
        y_val.append(money)
    length = len(y_val)


    result = 0



    for i in range(length-1):
        if y_val[i] < y_val[i+1]:
            result += y_val[i+1] - y_val[i]
    print(result)




