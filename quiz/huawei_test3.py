import sys

if __name__ == "__main__":
    lst = []
    # for line in sys.stdin:
    #     num = int(line.strip())
    #     if num != 0:
    #         lst.append(num)
    #     else:
    #         break
    #
    # for num in lst:
    #     # num = int(sys.stdin.readline().strip())
    #
    #     if num == 0:
    #         break
    #
    #     mod = 0
    #     total_drink = 0
    #     this_drink = 0
    #     while True:
    #         mod = num % 3
    #         this_drink = int(num / 3)
    #         num = this_drink + mod
    #         total_drink += this_drink
    #         if num < 3:
    #             break
    #     if num == 2:
    #         total_drink += 1
    #     print(total_drink)

    for line in sys.stdin:
        print(line)

