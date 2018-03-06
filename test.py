import numpy
import random
from torch.autograd import Variable, Function
import torch
import math
from torch import nn
import numpy as np

# random.seed(0)
# label = numpy.zeros((5,), dtype='int32')
# print label
# label[0:6] = 1
# print label
# a, b = 'a_b'.split('_')
# print a, b
# print random.random()

# data = Variable(torch.ones(10,5), requires_grad=True)
# a,b = data.data.topk(1)
# c = Variable(b)
# print data.data.topk(1)

def random_pick(some_list, probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item
# print random_pick([0,1], [0.5,0.5])

def schedule_samp_rate(iteration):
    k = 50
    rate = k / (k + math.exp(iteration / k))
    return rate

# print schedule_samp_rate(919)
# a = Variable(torch.zeros(2,3,4))
# b = Variable(torch.FloatTensor([[3,4,5],[66,3,4]]))
# print b
# c = b.unsqueeze(1).expand(2,3,3)
# print c
# s = nn.Softmax()
# print s(c.contiguous().view(-1, 3)).view(2,3,3)
# print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(0)))
# print(torch.backends.cudnn.version())
# from configurations import config, logger, to_np
# from batch_getter import BatchGetter
# import time
# from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingOutLayer
# emb = LoadEmbedding('res/emb.txt')
# print 'finish loading embedding'
# batch_getter = BatchGetter('data/train', 'PER_NAM')
# print 'finish loading train data'
# embedding_layer = EmbeddingLayer(emb)
# d = embedding_layer.get_out_dim()
# att_layer = AttentionFlowLayer(2 * d)
# model_out_layer = ModelingOutLayer(8*d, d, 2, 2)
# # models = [embedding_layer, att_layer, model_out_layer]
# # opts = [emb_opt, att_opt, model_out_opt]
#
# if config['USE_CUDA']:
#     att_layer.cuda(config['cuda_num'])
#     embedding_layer.cuda(config['cuda_num'])
#     model_out_layer.cuda(config['cuda_num'])
# rate = schedule_samp_rate(97)
# a = random_pick([0,1], [rate, 1 - rate])
# print a
# f = open('tt.txt','w')
# f.write('a')
# f.close()
# a = [1,2]
# b = [3,4]
# b.pop()
# a.extend(b)
# a[3] = 100
# print a,b
# import sys
# print sys.argv[0]+'ddd'
# import os
# os.system('cp {}/test {}/test1'.format('model1', 'model1'))
batch_size = 4
max_length = 3
hidden_size = 3
n_layers = 3
torch.manual_seed(0)
# container
batch_in = torch.zeros((batch_size, max_length, 1))

#data
vec_1 = torch.FloatTensor([[1, 2, 3]])
vec_2 = torch.FloatTensor([[1, 2, 0]])
vec_3 = torch.FloatTensor([[1, 0, 0]])

batch_in[0] = vec_1.transpose(0, 1)
batch_in[1] = vec_2.transpose(0, 1)
batch_in[2] = vec_3.transpose(0, 1)

batch_in = Variable(batch_in)
print(batch_in)

seq_lengths = [3,3,2,1]
# print batch_in
pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths, batch_first=True)
print(pack)
print(torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True))
# un_packed = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
# print un_packed
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True, bidirectional=True)
h0 = Variable(torch.randn(n_layers*2, batch_size, hidden_size))

#forward
out, _ = rnn(pack, h0)
print(out)
unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
print(unpacked, _)
# from conll_datagen.data_trans import BatchGetter
# a = BatchGetter('data/ttt', 'a', 1)
# a = 'I-MISC'
# a[1][0] = 'b'
# a = 'word'

# print a[0] == 'I'
# from conll_datagen.data_trans import p
# mask = torch.zeros(3,4,5)
# T_len = [1,2,3]
# J_len = [3, 2,1]
# batch = 0
# for t, j in zip(T_len, J_len):
#     mask[batch, :t, :j] = 1
#     batch += 1
# print mask
class GRL(Function):
    @staticmethod
    def forward(ctx, input):
        print('input:', input.get_device())
        a = input.clone()
        return a*2
    @staticmethod
    def backward(ctx, grad_output):
        print('output:', grad_output.get_device())
        return grad_output * -0.1


# class MyMod(nn.Module):
#     def forward(self, x):
#         return GRL()(x)

x = Variable(torch.ones(2, 2).cuda(1), requires_grad = True)
# x = x.cuda(1)
y = x ** 2
mymod = GRL.apply
z = mymod(y)
# z = y.sum()
z.backward(torch.ones(2, 2).cuda(1))
print("first backward of x is:")
# x.grad = x.grad * -0.1
print(x.grad)
# z.backward(2*torch.ones(2, 2))
# print("second backward of x is:")
# print(y.grad)

# m = nn.Sigmoid()
# loss = nn.BCELoss()
# input = Variable(torch.randn(3), requires_grad=True)
# target = Variable(torch.FloatTensor(3).random_(2))
# output = loss(m(input), target)
# output.backward()
# class MyFun(torch.autograd.Function):
#     def forward(self, inp):
#         return inp
#
#     def backward(self, grad_out):
#         grad_input = grad_out.clone()
#         print('Custom backward called!')
#         return grad_input
#
# # class MyMod(nn.Module):
# #     def forward(self, x):
# #         return MyFun()(x)
#
# mod = MyMod()
#
# y = Variable(torch.randn(1), requires_grad=True)
# z = mod(y)
# z.backward()
a = torch.FloatTensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

print(a)
print(torch.sum(a, 1))


