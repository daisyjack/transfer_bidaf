import torch
from torch.autograd import Variable
import torch.nn as nn
# from itertools import ifilter
import numpy
import sys

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.a = nn.Parameter(torch.ones(9))
        self.b = nn.Parameter(torch.ones(9))

    def forward(self):
        return 3*self.a + self.b

class N(nn.Module):
    def __init__(self):
        super(N, self).__init__()

    def forward(self, a):
        self.a = a
        return a



m = M()
n = N()

loss = n(m.a) + m()
loss = torch.sum(loss)
m_opt = torch.optim.SGD(m.parameters(), 0.001)
n_opt = torch.optim.SGD(n.parameters(), 0.001)
loss.backward()
m_opt.step()
n_opt.step()
pass
print([1,2][0:-1])

print([1] + [2])

mat1 = torch.randn(2, 3,3)
mat2 = torch.randn(3, 3)
print(torch.matmul(mat1, mat2))

# print m.a.grad
# a.sum().backward()
# o = torch.optim.Adadelta(ifilter(lambda p: p.requires_grad, m.parameters()))
# m.b.requires_grad = False
# for i in m.parameters():
#     print i.requires_grad, i.grad



# x = Variable(torch.ones(2, 2), requires_grad = True)
# y = x ** 2
# y.backward(torch.ones(2, 2))
# print "first backward of x is:"
# print x.grad
# y = x ** 2
# y.backward(2*torch.ones(2, 2))
# print "second backward of x is:"
# print x.grad
import string
table = str.maketrans({key: None for key in string.punctuation})
a = numpy.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a.translate(table))
print(a)
print(a[-1])
def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
a = "\\\"ee".split()
with open('data/ttt') as f:
    a = f.read()
    a = '  d  d  \n'
    print(a, a.split())

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--small_crf', action='store_false')
args = parser.parse_args()
print(args.small_crf)
a = torch.from_numpy(numpy.array([[1,2,3],[4,5,6],[7,8,9]])).type(torch.FloatTensor)
print(a)
torch.manual_seed(1)


class A(nn.Module):
    def __init__(self, p):
        super(A, self).__init__()
        self.d = nn.Dropout(p)
    def forward(self, a):
        return self.d(a)

class B(nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.aa = A(0.1)
        self.rnn = nn.GRU(input_size=10, hidden_size=5, num_layers=2, bidirectional=True,
                          dropout=0.1)
    def forward(self, a):
        return self.aa(a)

bb = B()
bb.eval()
bb.train()

print(eval('bb')(a))

def fun(*input):
    return input

# b, c = fun()
# print(' '.join('/person/e'.split('/')[1:]))


from fun.fun import get_cur_path1
print(get_cur_path1())
print(sys.path)
print('/other/body_part'.split('/'))
a = set([2,2,1,4])
a.update([2,3])
print(a)
a = numpy.zeros((2,1), dtype='int32')
a[0, :0] = 1
b = numpy.zeros((0,1), dtype='int32')
print(numpy.vstack((a, b)))
a = numpy.array([[0.6,0.4], [0.3,0.3]], dtype='float64')
print(a)
print((a>0.5).astype(int).tolist())
a = a[numpy.newaxis, ...]
soft = nn.Softmax()
# print(soft(a))
print('/person'.split('/'))
print(2*1e-2)
