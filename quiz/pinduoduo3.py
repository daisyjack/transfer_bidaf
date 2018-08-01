import sys



if __name__ == "__main__":
    # 读取第一行的n
    line = sys.stdin.readline().strip()
    values = list(map(int, line.split()))
    s1 = 0
    e1 = 0
    s2 = 0
    e2 = 0
    if values[0] + values[1] < values[2]:
        s1 = values[0]
        e1 = values[0] + values[1]
    elif values[0] + values[1] >= values[2]:
        if values[0] < values[2]:
            s1 = values[0]
            e1 =
        s1 = e1 = values[2]
        if
