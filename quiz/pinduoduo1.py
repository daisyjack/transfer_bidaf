#coding=utf-8
import sys

def cal_normal(cal):
    # cal.append('end')
    op_lst = ['*', '+']
    num_stack = []
    op_stack = []
    for a in cal:
        if a not in op_lst:
            num_stack.append(a)
        elif a in op_lst:
            stop = True
            while stop:
                if len(op_stack) == 0 or (a == '*' and op_stack[-1] == '+'):
                    op_stack.append(a)
                    stop = False
                else:
                    op = op_stack.pop()
                    num1 = int(num_stack.pop())
                    num0 = int(num_stack.pop())
                    if op == '+':
                        num_stack.append(num0 + num1)
                    else:
                        num_stack.append(num0 * num1)

                    # if a == '*' and op_stack[-1] == '+':
                    #     stop = False
        # elif a == 'end':
        #     op = op_stack.pop()
        #     num1 = int(num_stack.pop())
        #     num0 = int(num_stack.pop())
        #     if op == '+':
        #         num_stack.append(num0 + num1)
        #     else:
        #         num_stack.append(num0 * num1)
    if len(op_stack) == 0:
        return int(num_stack.pop())
    op = op_stack.pop()
    num1 = int(num_stack.pop())
    num0 = int(num_stack.pop())
    if op == '+':
        num_stack.append(num0 + num1)
    else:
        num_stack.append(num0 * num1)
    result = num_stack.pop()
    return result

def cal_left(cal):
    op_lst = ['*', '+']
    num_stack = []
    op_stack = []
    for a in cal:
        if a not in op_lst:
            num_stack.append(a)
        elif a in op_lst:
            if len(op_stack) == 0:
                op_stack.append(a)
            else:
                op = op_stack.pop()
                num1 = int(num_stack.pop())
                num0 = int(num_stack.pop())
                if op == '+':
                    num_stack.append(num0 + num1)
                else:
                    num_stack.append(num0 * num1)
                op_stack.append(a)
    if len(op_stack) == 0:
        return int(num_stack.pop())
    op = op_stack.pop()
    num1 = int(num_stack.pop())
    num0 = int(num_stack.pop())
    if op == '+':
        num_stack.append(num0 + num1)
    else:
        num_stack.append(num0 * num1)
    result = num_stack.pop()
    return result







if __name__ == "__main__":
    # 读取第一行的n
    cal = list(sys.stdin.readline().strip())
    n = int(sys.stdin.readline().strip())
    m = cal_normal(cal)
    l = cal_left(cal)
    if n == m and n == l:
        print('U')
    elif n == l:
        print('L')
    elif n == m:
        print('M')
    elif n != m and n != l:
        print('I')


