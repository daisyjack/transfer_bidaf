# coding: utf-8

from highway import Highway, HW
import torch
import torch.nn as nn
import codecs
from configurations import config, to_np
from batch_getter import BatchGetter, get_source_mask
from torch.autograd import Variable
import torch.nn.init
import torch.nn.functional as F
import utils

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

class LoadEmbedding(object):
    def __init__(self, emb_file):
        with codecs.open(emb_file, mode='rb', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    if i == 0:
                        parts = line.split(' ')
                        self.voc_size = int(parts[0]) + 8
                        self.emb_size = int(parts[1])
                        self.embedding_tensor = torch.zeros(self.voc_size, self.emb_size)
                    else:
                        parts = line.split(' ')
                        for j, part in enumerate(parts[1:]):
                            self.embedding_tensor[i+2, j] = float(part)

    def get_embedding_tensor(self):
        return self.embedding_tensor

    def get_voc_size(self):
        return self.voc_size

    def get_emb_size(self):
        return self.emb_size

def exp_mask(val, T_len, J_len):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked (B, T, J)
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    mask = Variable(torch.zeros(val.size()))
    if config['USE_CUDA']:
        mask = mask.cuda(val.get_device())
    batch = 0
    for t, j in zip(T_len, J_len):
        mask[batch, :t, :j] = 1
        batch += 1
    return val + (1 - mask) * VERY_NEGATIVE_NUMBER

def exp_mask_2d(val, T_len):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked (B, T, J)
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    mask = Variable(torch.zeros(val.size()))
    if config['USE_CUDA']:
        mask = mask.cuda(val.get_device())
    batch = 0
    for t in T_len:
        mask[batch, :t] = 1
        batch += 1
    return val + (1 - mask) * VERY_NEGATIVE_NUMBER


class EmbeddingLayer(nn.Module):
    # self.out_dim: d
    def __init__(self, loaded_embedding, dropout_p=config['dropout']):
        super(EmbeddingLayer, self).__init__()
        self.loaded_embedding = loaded_embedding
        self.dropout = nn.Dropout(dropout_p)
        # self.onto = onto

        self.out_dim = 0
        self.input_dim = 0
        word_emb_size = self.loaded_embedding.get_emb_size()
        self.word_emb_size = word_emb_size
        self.word_embedding = nn.Embedding(self.loaded_embedding.get_voc_size(), word_emb_size)
        self.word_embedding.weight.data.copy_(self.loaded_embedding.get_embedding_tensor())


        # self.word_embedding.weight.requires_grad = False
        self.input_dim += 1
        self.out_dim += self.loaded_embedding.get_emb_size()
        self._use_gaz = config['use_gaz']

        self.gemb = None
        if self._use_gaz:
            gazsize = len(config['Gazetteers'])
            gazdim = config['gaz_emb_dim']
            self.gazsize = gazsize
            self.gemb = nn.Linear(gazsize, gazdim, bias=False)
            torch.nn.init.uniform(self.gemb.weight, -config['weight_scale'], config['weight_scale'])
            self.input_dim += gazsize
            self.out_dim += gazdim

        self._use_char_conv = config['use_char_conv']
        self.char_emb = None
        self.char_conv = None
        if self._use_char_conv:
            vocChar = config['CharVoc']
            maxCharLen = config['max_char']
            self.char_len = maxCharLen
            char_emb_size = config['char_emb_dim']
            self.char_emb = nn.Embedding(vocChar.getVocSize(), char_emb_size)
            torch.nn.init.uniform(self.char_emb.weight, -config['weight_scale'], config['weight_scale'])
            self.input_dim += maxCharLen * 2
            self.char_conv = nn.Conv1d(in_channels=char_emb_size, out_channels=config['out_channel_dims'],
                                       kernel_size=config['filter_size'], stride=1, padding=2)
            utils.init_weight(self.char_conv.weight)
            # torch.nn.init.uniform(self.char_conv.weight, -config['weight_scale'], config['weight_scale'])
            torch.nn.init.constant(self.char_conv.bias, 0)
            self.out_dim += config['out_channel_dims']
            self.conv_active = nn.ReLU()
        # self.highway = Highway(self.out_dim, config['highway_num_layers'], F.relu, dropout_p)
        self.c_hw = HW(self.out_dim, config['highway_num_layers'], dropout_p)
        self.q_hw = HW(self.out_dim, config['highway_num_layers'], dropout_p)
        self.c_hw.rand_init()
        self.q_hw.rand_init()
        self.encode_rnn = nn.GRU(input_size=self.out_dim, hidden_size=config['hidden_size'], num_layers=1, bidirectional=True, dropout=dropout_p)
        # else:
        #     self.encode_rnn = nn.GRU(input_size=self.out_dim, hidden_size=config['hidden_size'], num_layers=1,
        #                              bidirectional=True)
        utils.init_rnn(self.encode_rnn)
        # for name, param in self.question_trans.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.uniform(param, -config['weight_scale'], config['weight_scale'])


    # input: (batch, seq_length, 51) h_0: (num_layers(1) * num_directions(2), batch, d)
    def forward(self, input, h_0, seq_length, step=0, onto_emb=None, name='Q'):
        input_size = input.size()

        outs = []
        word_slice = input[:, :, 0]
        if name == 'Q' and config['label_emb']:
            word_emb = onto_emb(word_slice)
        else:
            word_emb = self.word_embedding(word_slice)  # (batch, seq_length, word_emb)
        # word_emb = self.word_embedding(word_slice)
        outs.append(word_emb)
        curr_end = 1

        if self._use_gaz:
            gazStart = curr_end
            gazEnd = gazStart + self.gazsize
            if config['USE_CUDA']:
                a = input[:, :, gazStart:gazEnd]
                b = a.type(torch.cuda.FloatTensor)
                c = b.view(-1, self.gazsize)
                d = self.gemb(c)
                e = d.view(input_size[0], input_size[1], -1)
                f = e.contiguous()
                outs.append(f)
                # outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.cuda.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            else:
                outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            curr_end = gazEnd

        if self._use_char_conv:
            # print(input.size(), input.get_device())
            # print(curr_end, curr_end + self.char_len)
            chars = input[:, :, curr_end:curr_end + self.char_len].contiguous()
            chars_mask = input[:, :, (curr_end + self.char_len):(curr_end + 2 * self.char_len)]
            if config['USE_CUDA']:
                chars_mask = chars_mask.type(torch.cuda.FloatTensor)
            else:
                chars_mask = chars_mask.type(torch.FloatTensor)
            chars_size = chars.size()
            char_view = chars.view(-1, self.char_len)  # (B*seq_length, max_char_len)
            char_emb_out = self.char_emb(char_view)  # (B*seq_length, max_char_len, char_emb)
            chars_mask = chars_mask.view(-1, self.char_len)  # (B*seq_length, max_char_len)
            char_emb_out = char_emb_out * chars_mask.unsqueeze(2).expand_as(char_emb_out)


            # char_shape = char_emb_out.shape
            # char_emb_out = char_emb_out.reshape((char_shape[0] * char_shape[1], char_shape[2], 1, char_shape[3]))
            # char_conv_out = self.char_conv.apply(char_emb_out)
            # char_conv_out = self.conv_active.apply(char_conv_out)
            # char_conv_out = char_conv_out.reshape(char_shape)
            # char_conv_out = char_conv_out * chars_mask.dimshuffle(0, 1, 2, 'x')
            # char_conv_out = tensor.max(char_conv_out, axis=2)

            char_emb_out = char_emb_out.transpose(1, 2)  # (B*seq_length, char_emb, char_len)
            # if config['use_dropout'] and (not config['freeze']):
            char_emb_out = self.dropout(char_emb_out)  # (B*seq_length, char_emb, char_len)
            char_conv_out = self.char_conv(char_emb_out)  # (B*seq_length, out_channel, char_len)
            char_conv_out = self.conv_active(char_conv_out)
            char_conv_out = char_conv_out.transpose(1, 2)  # (B*seq_length, char_len, out_channel)
            char_conv_out = char_conv_out * chars_mask.unsqueeze(2).expand_as(char_conv_out)  # (B*seq_length, char_len, out_channel)
            char_conv_out, _ = torch.max(char_conv_out, 1)
            char_conv_out = char_conv_out.view(chars_size[0], chars_size[1], -1)  # (B, seq_length, out_channel)
            char_conv_out = self.dropout(char_conv_out)
            outs.append(char_conv_out)
        output = torch.cat(outs, dim=-1)
        mask = Variable(get_source_mask(input_size[0], self.out_dim, input_size[1], seq_length))
        if config['USE_CUDA']:
            mask = mask.cuda(input.get_device())
        mask = mask.transpose(0, 1)
        embedded = output * mask  # embedded: (batch, seq_length, emb_size)
        embedded = embedded.view(-1, self.out_dim).contiguous()
        if name == 'C':
            embedded = self.c_hw(embedded)
        elif name == 'Q':
            embedded = self.q_hw(embedded)
        embedded = embedded.view(input_size[0], input_size[1], -1).contiguous()
        embedded = embedded * mask
        embedded = embedded.transpose(0, 1).contiguous()  # embedded: (seq_length, batch, emb_size)
        rnn_output, h_n = self.encode_rnn(embedded, h_0)
        rnn_mask = Variable(get_source_mask(input_size[0], config['hidden_size']*2, input_size[1], seq_length))
        if config['USE_CUDA']:
            rnn_mask = rnn_mask.cuda(input.get_device())
        rnn_output = rnn_output * rnn_mask
        # if config['use_dropout']:
        #     rnn_output = self.dropout(rnn_output)
        # logger.histo_summary('EmbeddingLayer/output', to_np(rnn_output), step)
        return rnn_output  # (seq_len, batch, hidden_size(d=100) * num_directions(2))

    def get_out_dim(self):
        return self.out_dim

class QLabel(nn.Module):
    def __init__(self, loaded_embedding, dropout_p=config['dropout']):
        super(QLabel, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.loaded_embedding = loaded_embedding
        self.dropout = nn.Dropout(dropout_p)

        self.out_dim = 0
        word_emb_size = self.loaded_embedding.get_emb_size()
        self.word_emb_size = word_emb_size
        self.word_embedding = nn.Embedding(self.loaded_embedding.get_voc_size(), word_emb_size)
        self.word_embedding.weight.data.copy_(self.loaded_embedding.get_embedding_tensor())
        self.word_embedding.weight.requires_grad = False

        self.out_dim += self.loaded_embedding.get_emb_size()
        self.linear = nn.Linear(self.word_emb_size, 2*config['hidden_size'])
        utils.init_linear(self.linear)


    # input: (batch, seq_length, 51) h_0: (num_layers(1) * num_directions(2), batch, d)
    def forward(self, input, h_0, seq_length, step=0, name='Q'):
        input_size = input.size()
        word_slice = input[:, :, 0]
        word_emb = self.word_embedding(word_slice)  # (batch, seq_length, word_emb)
        trans_emb = nn.functional.relu(self.linear(word_emb))
        out = self.dropout(trans_emb).transpose(0, 1)  # (seq_length, batch, 2*d)
        mask = Variable(get_source_mask(input_size[0], config['hidden_size'] * 2, input_size[1], seq_length))
        if config['USE_CUDA']:
            mask = mask.cuda(input.get_device())
        out = out * mask
        return out

    def get_out_dim(self):
        return self.out_dim




class QEmbeddingLayer(nn.Module):
    # self.out_dim: d
    def __init__(self, word_emb_size, dropout_p=config['dropout']):
        super(QEmbeddingLayer, self).__init__()
        # self.loaded_embedding = loaded_embedding
        self.dropout = nn.Dropout(dropout_p)

        self.out_dim = 0
        self.input_dim = 0
        # word_emb_size = self.loaded_embedding.get_emb_size()
        # self.word_embedding = nn.Embedding(self.loaded_embedding.get_voc_size(), word_emb_size)
        # self.word_embedding.weight.data.copy_(self.loaded_embedding.get_embedding_tensor())
        # self.word_embedding.weight.requires_grad = False
        self.input_dim += 1
        self.out_dim += word_emb_size
        self._use_gaz = config['use_gaz']

        self.gemb = None
        if self._use_gaz:
            gazsize = len(config['Gazetteers'])
            gazdim = config['gaz_emb_dim']
            self.gazsize = gazsize
            self.input_dim += gazsize
            self.out_dim += gazdim

        self._use_char_conv = config['use_char_conv']
        self.char_emb = None
        self.char_conv = None
        if self._use_char_conv:
            vocChar = config['CharVoc']
            maxCharLen = config['max_char']
            self.char_len = maxCharLen
            char_emb_size = config['char_emb_dim']
            self.input_dim += maxCharLen * 2
            self.out_dim += config['out_channel_dims']
            self.conv_active = nn.ReLU()
        # self.encode_rnn = nn.GRU(input_size=self.out_dim, hidden_size=self.out_dim, num_layers=1, bidirectional=True)
        # for name, param in self.encode_rnn.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.orthogonal(param)
        self.question_trans = nn.Linear(self.out_dim, self.out_dim*2)
        for name, param in self.question_trans.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0)
            elif 'weight' in name:
                torch.nn.init.uniform(param, -config['weight_scale'], config['weight_scale'])

    # input: (batch, seq_length, 51) h_0: (num_layers(1) * num_directions(2), batch, d)
    def forward(self, input, h_0, seq_length, word_embedding, gemb, char_emb, char_conv, highway, step=0, name='Q'):
        input_size = input.size()

        outs = []
        word_slice = input[:, :, 0]
        word_emb = word_embedding(word_slice)  # (batch, seq_length, word_emb)
        outs.append(word_emb)
        curr_end = 1

        if self._use_gaz:
            gazStart = curr_end
            gazEnd = gazStart + self.gazsize
            if config['USE_CUDA']:
                a = input[:, :, gazStart:gazEnd]
                b = a.type(torch.cuda.FloatTensor)
                c = b.view(-1, self.gazsize)
                d = gemb(c)
                e = d.view(input_size[0], input_size[1], -1)
                f = e.contiguous()
                outs.append(f)
                # outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.cuda.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            else:
                outs.append(self.gemb(input[:, :, gazStart:gazEnd].type(torch.FloatTensor).view(-1, self.gazsize)).view(input_size[0], input_size[1], -1).contiguous())
            curr_end = gazEnd

        if self._use_char_conv:
            chars = input[:, :, curr_end:curr_end + self.char_len].contiguous()
            chars_mask = input[:, :, (curr_end + self.char_len):(curr_end + 2 * self.char_len)]
            if config['USE_CUDA']:
                chars_mask = chars_mask.type(torch.cuda.FloatTensor)
            else:
                chars_mask = chars_mask.type(torch.FloatTensor)
            chars_size = chars.size()
            char_view = chars.view(-1, self.char_len)  # (B*seq_length, max_char_len)
            char_emb_out = char_emb(char_view)  # (B*seq_length, max_char_len, char_emb)
            chars_mask = chars_mask.view(-1, self.char_len)  # (B*seq_length, max_char_len)
            char_emb_out = char_emb_out * chars_mask.unsqueeze(2).expand_as(char_emb_out)


            # char_shape = char_emb_out.shape
            # char_emb_out = char_emb_out.reshape((char_shape[0] * char_shape[1], char_shape[2], 1, char_shape[3]))
            # char_conv_out = self.char_conv.apply(char_emb_out)
            # char_conv_out = self.conv_active.apply(char_conv_out)
            # char_conv_out = char_conv_out.reshape(char_shape)
            # char_conv_out = char_conv_out * chars_mask.dimshuffle(0, 1, 2, 'x')
            # char_conv_out = tensor.max(char_conv_out, axis=2)

            char_emb_out = char_emb_out.transpose(1, 2)  # (B*seq_length, char_emb, char_len)
            char_conv_out = char_conv(char_emb_out)  # (B*seq_length, out_channel, char_len)
            char_conv_out = self.conv_active(char_conv_out)
            char_conv_out = char_conv_out.transpose(1, 2)  # (B*seq_length, char_len, out_channel)
            char_conv_out = char_conv_out * chars_mask.unsqueeze(2).expand_as(char_conv_out)  # (B*seq_length, char_len, out_channel)
            char_conv_out, _ = torch.max(char_conv_out, 1)
            char_conv_out = char_conv_out.view(chars_size[0], chars_size[1], -1)  # (B, seq_length, out_channel)
            outs.append(char_conv_out)
        output = torch.cat(outs, dim=-1)
        mask = Variable(get_source_mask(input_size[0], self.out_dim, input_size[1], seq_length))
        if config['USE_CUDA']:
            mask = mask.cuda(input.get_device())
        mask = mask.transpose(0, 1)
        embedded = output * mask  # embedded: (batch, seq_length, emb_size)
        embedded = embedded.view(-1, self.out_dim).contiguous()
        embedded = highway(embedded)
        embedded = embedded.view(input_size[0], input_size[1], -1).contiguous()
        embedded = embedded * mask
        embedded = embedded.transpose(0, 1).contiguous()  # embedded: (seq_length, batch, emb_size)
        embedded = self.question_trans(embedded)
        # rnn_output, h_n = self.encode_rnn(embedded, h_0)
        rnn_mask = Variable(get_source_mask(input_size[0], self.out_dim*2, input_size[1], seq_length))
        if config['USE_CUDA']:
            rnn_mask = rnn_mask.cuda(input.get_device())
        rnn_output = embedded * rnn_mask
        if config['use_dropout']:
            rnn_output = self.dropout(rnn_output)
        # logger.histo_summary('EmbeddingLayer/output', to_np(rnn_output), step)
        return rnn_output  # (seq_len, batch, hidden_size(100+100=d) * num_directions(2))

    def get_out_dim(self):
        return self.out_dim

class AttentionFlowLayer(nn.Module):
    # emb_size: 2d
    def __init__(self, emb_size):
        super(AttentionFlowLayer, self).__init__()
        self.att_w = nn.Parameter(torch.FloatTensor(3 * emb_size, 1))
        # torch.nn.init.uniform(self.att_w, -config['weight_scale'], config['weight_scale'])
        utils.init_weight(self.att_w)
        self.softmax = nn.Softmax()
        if config['gate']:
            self.gate_weight = nn.Parameter(torch.FloatTensor(4*emb_size, 4*emb_size))
            torch.nn.init.uniform(self.gate_weight, -config['weight_scale'], config['weight_scale'])


    # H, context:(T, batch, 2d) U, question: (J, batch, 2d)
    def forward(self, context, question, con_lens, qu_lens, step=0):
        context_t = context.transpose(0, 1).contiguous()  # (batch, T, 2d)
        context_size = context.size()
        question_size = question.size()
        context_len = context_size[0]
        question_len = question_size[0]
        S = Variable(torch.zeros(context_size[1], context_len, question_len))  # (batch, T, J)
        if config['USE_CUDA']:
            S = S.cuda(context.get_device())
        for t in range(context_len):
            for j in range(question_len):
                c = context[t, :, :]  # (batch, 2d)
                q = question[j, :, :]  # (batch, 2d)
                cat = torch.cat([c, q, c * q], dim=-1)
                att = torch.mm(cat, self.att_w)  # (batch, 1)
                S[:, t, j] = att
        S = exp_mask(S, con_lens, qu_lens)
        if config['sigmoid']:
            c2q_att_w = torch.sigmoid(S.view(-1, question_len).contiguous()).view(context_size[1],
                                                                                  context_len, question_len).contiguous()  # (batch, T, J)
        else:
            c2q_att_w = self.softmax(S.view(-1, question_len).contiguous()).view(context_size[1],
                                                                                 context_len, question_len).contiguous() # (batch, T, J)

        c2q_att = torch.bmm(c2q_att_w, question.transpose(0, 1).contiguous())  # U~： (batch, T, 2d)
        value, index = torch.max(S, 2)  # value(batch, T)
        # if config['sigmoid']:
        #     value = torch.sigmoid(value)
        # else:
        value = self.softmax(value)
        value = value.unsqueeze(1).expand(context_size[1], context_size[0], context_size[0])  # value(batch, T, T)
        q2c_att = torch.bmm(value, context_t)  # H~: (batch, T, 2d)
        G = torch.cat([context_t, c2q_att, context_t * c2q_att, context_t * q2c_att], dim=2)
        mask = Variable(get_source_mask(context_size[1], context_size[2]*4, context_size[0], con_lens))
        mask = mask.transpose(0, 1)
        if config['USE_CUDA']:
            mask = mask.cuda(context.get_device())
        G = G * mask
        # logger.histo_summary('AttentionFlowLayer/output', to_np(G), step)
        if config['gate']:
            gate = torch.matmul(G, self.gate_weight)
            gate = torch.sigmoid(gate)
            G = gate * G
        return G  # (batch, T, 8d)

    # def get_att_mask(self, batch_num, con_lens, qu_lens):



class ModelingLayer(nn.Module):
    # input_size: 8d  hidden_size: d  num_layers: 2
    def __init__(self, input_size, hidden_size, num_layers, dropout=config['dropout']):
        super(ModelingLayer, self).__init__()
        self.num_layers = num_layers
        # if config['use_dropout']:
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout, bidirectional=True)
        # else:
        #     self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        #                       bidirectional=True)
        # for name, param in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.orthogonal(param)
        utils.init_rnn(self.rnn)
        self.dropout = nn.Dropout(dropout)
        # self.W = nn.Parameter(torch.FloatTensor(hidden_size * 10, class_num))


    # h_0: (num_layers * num_directions, batch, d) G: (batch, T, 8d)
    def forward(self, h_0, G, con_lens, step=0):
        G_t = G.transpose(0, 1).contiguous()
        M, h_n = self.rnn(G_t, h_0)  # M: (T, batch, 2d)
        # logger.histo_summary('ModelingOutLayer/rnn_output', to_np(M), step)
        size = M.size()
        mask = Variable(get_source_mask(size[1], size[2], size[0], con_lens))
        if config['USE_CUDA']:
            mask = mask.cuda(M.get_device())
        M = M * mask
        M = self.dropout(M)
        return M.transpose(0, 1).contiguous()  # M: (batch, T, 2d)
        # cat = torch.cat([G, M.transpose(0, 1).contiguous()], 2)  # cat: (batch, T, 10d)
        # prob = self.W(cat)  # (batch, T, 5)
        # # prob_size = prob.size()
        # # prob = prob.view(-1, self.class_num).contiguous()
        # # prob = self.softmax(prob)
        # # prob = prob.view(-1, prob_size[1], self.class_num)  # (batch, T, 2)
        # return prob


class StartProbLayer(nn.Module):
    def __init__(self, hidden_size, dropout=config['dropout']):
        """
        :param hidden_size: 10*d
        """
        super(StartProbLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(hidden_size, 1)
        # for name, param in self.W.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.uniform(param, -config['weight_scale'], config['weight_scale'])
        utils.init_linear(self.W)
        self.logSoftmax = nn.LogSoftmax()

    def forward(self, M, G, con_lens):
        """
        :param M: (batch, T, 2d)
        :param G: (batch, T, 8d)
        :return: (batch, T)
        """
        cat = torch.cat([G, M], 2)  # cat: (batch, T, 10d)
        logits = self.W(cat)  # (batch, T, 1)
        logits = self.dropout(logits)
        logits = logits.squeeze(2)
        logits = exp_mask_2d(logits, con_lens)
        probs = self.logSoftmax(logits)  # (batch, T)
        return probs


class EndProbLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=config['dropout']):
        """
        :param input_size: 2d
        :param hidden_size: d
        """
        super(EndProbLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # if config['use_dropout']:
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True,
                              dropout=dropout)
        # else:
        #     self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        # for name, param in self.rnn.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.orthogonal(param)
        utils.init_rnn(self.rnn)
        self.W = nn.Linear(10*hidden_size, 1)
        # for name, param in self.W.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.uniform(param, -config['weight_scale'], config['weight_scale'])
        utils.init_linear(self.W)
        self.logSoftmax = nn.LogSoftmax()
        self.num_layers = 1

    def forward(self, M, G, h_0, con_lens):
        """
        :param M: (batch, T, 2d)
        :param G: (batch, T, 8d)
        :param h_0: (num_layers * num_directions, batch, hidden_size)
        :return:  (batch, T)
        """
        M_t = M.transpose(0, 1)
        M_2, h_n = self.rnn(M_t, h_0)  # M_2:(T, batch, 2d)
        size = M_2.size()
        mask = Variable(get_source_mask(size[1], size[2], size[0], con_lens))
        if config['USE_CUDA']:
            mask = mask.cuda(M.get_device())
        M_2 = M_2 * mask
        M_2 = self.dropout(M_2)
        cat = torch.cat([G, M_2.transpose(0, 1)], 2)  # cat: (batch, T, 10d)
        logits = self.W(cat)  # (batch, T, 1)
        logits = self.dropout(logits)
        logits = logits.squeeze(2)
        logits = exp_mask_2d(logits, con_lens)
        probs = self.logSoftmax(logits)  # (batch, T)
        return probs




class NerOutLayer(nn.Module):
    def __init__(self, input_size, class_num, dropout=config['dropout']):
        """
        :param input_size: 10d
        :param class_num: BIOES PADDING START
        """
        super(NerOutLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)

        if config['large_crf']:
            self.W = nn.Linear(input_size, class_num*class_num)
        else:
            self.W = nn.Linear(input_size, class_num)
        utils.init_linear(self.W)
        # for name, param in self.W.named_parameters():
        #     if 'bias' in name:
        #         torch.nn.init.constant(param, 0)
        #     elif 'weight' in name:
        #         torch.nn.init.uniform(param, -config['weight_scale'], config['weight_scale'])

    def forward(self, M, G, con_lens):
        """
        :param M: (batch, T, 2d)
        :param G: (batch, T, 8d)
        :return: (batch, T, class_num)
        """

        cat = torch.cat([G, M], 2)
        logits = self.W(cat)  # (batch, T, class_num)
        self.dropout(logits)
        return logits

class NerHighway(nn.Module):
    def __init__(self, M_dim, G_dim, num_layers, dropout=config['dropout']):
        """

        :param M_dim: 2d
        :param G_dim: 8d
        :param dropout:
        """
        super(NerHighway, self).__init__()
        self.M_hw = HW(M_dim, num_layers, dropout)
        self.M_hw.rand_init()
        self.G_hw = HW(G_dim, num_layers, dropout)
        self.G_hw.rand_init()
        self.dropout = nn.Dropout(dropout)

    def forward(self, M, G):
        """

        :param M: (batch, T, 2d)
        :param G: (batch, T, 8d)
        :return:
        """
        M_trans = self.M_hw(M)
        M_trans = self.dropout(M_trans)
        G_trans = self.G_hw(G)
        G_trans = self.dropout(G_trans)
        return M_trans, G_trans






















































