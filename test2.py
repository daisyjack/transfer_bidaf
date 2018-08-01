#coding=utf-8
import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ans = 0
    lst = []
    for i in range(n):
        # 读取每一行1
        for j in range(2):
            line = sys.stdin.readline().strip()
            if j == 0:
                len = int(line)
            else:
                values = list(line)
        lst.append(values)
    for one in lst:

        val = ['X']
        val.extend(one)
        val.append('X')
        total = 0
        count = 0

        for pos, b in enumerate(val):

            if b == 'X' and pos == 0:
                pass
            elif b == '.':
                count += 1
            elif b == 'X' and pos != 0:
                count = count - 2
                if count == -1:
                    count = 1
                elif count == 0:
                    count = 1
                elif count < -1:
                    count = 0
                total += count
                count = 0
        print(total)



