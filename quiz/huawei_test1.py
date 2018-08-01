# coding: utf-8
import sys


if __name__ == "__main__":
    # 读取第一行的n
    while True:
        try:
            cal = int(sys.stdin.readline().strip())
            lst = []
            for i in range(cal):
                lst.append(int(sys.stdin.readline().strip()))
            a = list(set(lst))
            a.sort()
            for i in a:
                print(i)
        except:
            break
