import sys



if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    line = sys.stdin.readline().strip()
    values = list(map(int, line.split()))
    values.sort(reverse=True)
    taxi = 0
    blank = []
    for a in values:
        if a == 4:
            taxi += 1
        else:
            get = False
            for idx, b in enumerate(blank):
                if a <= b:
                    get = True
                    break
            if get:
                pop_num = blank.pop(idx)
                if pop_num > a:
                    blank.append(pop_num - a)
            else:
                taxi += 1
                blank.append(4 - a)

    print(taxi)



