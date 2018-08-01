import sys

if __name__ == "__main__":
    while True:
        try:
            cal = sys.stdin.readline().strip()

            print(int(cal, 16))
        except:
            break