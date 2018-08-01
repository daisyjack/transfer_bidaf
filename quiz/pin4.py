# coding: utf-8

import sys



if __name__ == "__main__":
    # 读取第一行的n
    max_l, liang = list(map(int, sys.stdin.readline().strip().split(' ')))
    num = sys.stdin.readline().strip()
    print(max_l, liang)
    print(num)
    min_num = '9'*max_l
    min_money = 1e10
    for can in range(0, 10):
        min_liang = []
        for i, a in enumerate(num):
            money = abs(int(a)-can)
            if len(min_liang) == 0:
                min_liang.append((money, i))
            else:
                app = False
                for j in range(0, len(min_liang)):
                    k = min_liang[j]
                    if money < k[0]:
                        min_liang.insert(j, (money, i))
                        break
                    elif money == k[0]:
                        if num[i] < num[k[1]]:
                            min_liang[j] = (money, i)
                        elif num[i] == num[k[1]]:
                            min_liang.insert(j, (money, i))
                        break
                    else:
                        app = True
                        break
                if app:
                    min_liang.append((money, i))
        b_min_liang = min_liang[0:liang]
        this_money = 0
        for b in b_min_liang:
            this_money += b[0]
            this_num = list(num)
            this_num[b[1]] = str(can)
        if this_money < min_money:
            min_money = this_money
            min_num = this_num
        elif this_money == min_money:
            if this_num < min_num:
                min_money = this_money
                min_num = this_num
    print(min_money)
    print(''.join(min_num))







