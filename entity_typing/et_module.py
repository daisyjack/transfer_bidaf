import torch
from torch import nn
from configurations import fg_config
import utils
from torch.autograd import Variable
from batch_getter import get_source_mask

class EmbeddingLayer(nn.Module):
    def __init__(self, loaded_embedding, dropout_p=fg_config['dropout']):
        super(EmbeddingLayer, self).__init__()
        self.loaded_embedding = loaded_embedding
        self.dropout = nn.Dropout(dropout_p)
        word_emb_size = self.loaded_embedding.get_emb_size()
        self.word_emb_size = word_emb_size
        self.word_embedding = nn.Embedding(self.loaded_embedding.get_voc_size(), word_emb_size, padding_idx=0)
        self.word_embedding.weight.data.copy_(self.loaded_embedding.get_embedding_tensor())
        for param in self.word_embedding.parameters():
            param.requires_grad = False

    def forward(self, input):
        """
        args:
        :param input: (B, S, 1) / (B, 1)
        :return:
        """
        input = input.squeeze(-1)
        output = self.word_embedding(input)  # (B, S, word_emb) / (B, word_emb)
        output = self.dropout(output)
        return output  # (B, S, word_emb) / (B, word_emb)


class CtxLSTM(nn.Module):
    def __init__(self, word_emb_size, dropout_p=fg_config['dropout']):
        super(CtxLSTM, self).__init__()
        self.word_emb_size = word_emb_size
        self.l_lstm = nn.LSTM(self.word_emb_size, fg_config['hidden_size'], fg_config['lstm_layers'], dropout=dropout_p, bidirectional=True)
        self.r_lstm = nn.LSTM(self.word_emb_size, fg_config['hidden_size'], fg_config['lstm_layers'], dropout=dropout_p, bidirectional=True)
        utils.init_rnn(self.l_lstm)
        utils.init_rnn(self.r_lstm)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, l_ctx_emb, r_ctx_emb, l_ctx_lens, r_ctx_lens):
        """
        args:
            l_ctx_emb: (B, S, word_emb)
            r_ctx_emb: (B, S, word_emb)
            l_ctx_lens: list
            r_ctx_lens: list
        :return:
        """
        batch_size = l_ctx_emb.size(0)
        h_0 = Variable(torch.zeros(fg_config['lstm_layers']*2, batch_size, fg_config['hidden_size']))
        c_0 = Variable(torch.zeros(fg_config['lstm_layers']*2, batch_size, fg_config['hidden_size']))
        if fg_config['USE_CUDA']:
            h_0 = h_0.cuda(fg_config['cuda_num'])
            c_0 = c_0.cuda(fg_config['cuda_num'])
        # (S, B, hidden_size*2)
        l_ctx_lstm, _ = self.l_lstm(l_ctx_emb.transpose(0, 1), (h_0, c_0))
        # (S, B, hidden_size*2)
        r_ctx_lstm, _ = self.r_lstm(r_ctx_emb.transpose(0, 1), (h_0, c_0))
        l_mask = Variable(get_source_mask(batch_size, fg_config['hidden_size']*2, fg_config['ctx_window_size'], l_ctx_lens))
        r_mask = Variable(get_source_mask(batch_size, fg_config['hidden_size']*2, fg_config['ctx_window_size'], r_ctx_lens))
        if fg_config['USE_CUDA']:
            l_mask = l_mask.cuda(fg_config['cuda_num'])
            r_mask = r_mask.cuda(fg_config['cuda_num'])
        l_ctx_lstm = l_ctx_lstm * l_mask
        r_ctx_lstm = r_ctx_lstm * r_mask
        l_ctx_lstm = self.dropout(l_ctx_lstm)
        r_ctx_lstm = self.dropout(r_ctx_lstm)
        # (S, B, hidden_size*2)
        return l_ctx_lstm, r_ctx_lstm


class CtxAtt(nn.Module):
    def __init__(self, hidden_size, word_emb_size):
        super(CtxAtt, self).__init__()
        self.hidden_size = hidden_size
        self.word_emb_size = word_emb_size
        self.att_weight = nn.Parameter(torch.FloatTensor(hidden_size*2, word_emb_size))
        utils.init_weight(self.att_weight)
        self.softmax = nn.Softmax()

    def forward(self, l_ctx_lstm, r_ctx_lstm, types_emb, mentions_emb, l_ctx_lens, r_ctx_lens, men_lens):
        """
        get mention representation and context representation

        args:
            l_ctx_lstm: (S, B, hidden_size*2)
            r_ctx_lstm: (S, B, hidden_size*2)
            types_emb: (B, word_emb)
            mentions_emb: (B, S, word_emb)
            l_ctx_lens: list
            r_ctx_lens: list
            men_lens: list
        :return:
        """
        l_ctx_lstm = l_ctx_lstm.transpose(0, 1).contiguous()  # (B, S, hidden_size*2)
        r_ctx_lstm = r_ctx_lstm.transpose(0, 1).contiguous()  # (B, S, hidden_size*2)
        tmp = torch.mm(l_ctx_lstm.view(-1, l_ctx_lstm.size(2)).contiguous(), self.att_weight)  # (B*S, word_emb)
        tmp = tmp.view(l_ctx_lstm.size(0), l_ctx_lstm.size(1), -1)  # (B, S, word_emb)
        l_weights = torch.bmm(tmp, types_emb.unsqueeze(2)).view(l_ctx_lstm.size(0), l_ctx_lstm.size(1))  # (B, S)
        l_weights = CtxAtt.exp_mask(l_weights, l_ctx_lens)  # (B, S)
        l_weights = self.softmax(l_weights).unsqueeze(2).expand_as(l_ctx_lstm)  # (B, S, hidden_size*2)
        l_ctx = torch.sum(l_weights * l_ctx_lstm, 1)  # (B, hidden_size*2)

        tmp = torch.mm(r_ctx_lstm.view(-1, r_ctx_lstm.size(2)).contiguous(), self.att_weight)  # (B*S, word_emb)
        tmp = tmp.view(l_ctx_lstm.size(0), l_ctx_lstm.size(1), -1)  # (B, S, word_emb)
        r_weights = torch.bmm(tmp, types_emb.unsqueeze(2)).view(l_ctx_lstm.size(0), l_ctx_lstm.size(1))  # (B, S)
        r_weights = CtxAtt.exp_mask(r_weights, r_ctx_lens)  # (B, S)
        r_weights = self.softmax(r_weights).unsqueeze(2).expand_as(r_ctx_lstm)  # (B, S, hidden_size*2)
        r_ctx = torch.sum(r_weights * r_ctx_lstm, 1)  # (B, hidden_size*2)

        ctx_rep = l_ctx + r_ctx  # (B, hidden_size*2)

        men_rep = torch.sum(mentions_emb, 1)  # (B, word_emb)
        men_len_mask = Variable(torch.ones(l_ctx_lstm.size(0), 1))
        if fg_config['USE_CUDA']:
            men_len_mask = men_len_mask.cuda(l_ctx_lstm.get_device())
        for b, l in enumerate(men_lens):
            men_len_mask[b, 0] = l
        men_len_mask = men_len_mask.expand_as(men_rep)
        men_rep = men_rep / men_len_mask

        # ctx_rep: (B, hidden_size*2), men_rep: (B, word_emb)
        return ctx_rep, men_rep


    @staticmethod
    def exp_mask(val, lens):
        VERY_BIG_NUMBER = 1e30
        VERY_SMALL_NUMBER = 1e-30
        VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
        VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
        mask = Variable(torch.zeros(val.size()))
        if fg_config['USE_CUDA']:
            mask = mask.cuda(val.get_device())
        batch = 0
        for i in lens:
            if i > 0:
                mask[batch, :i] = 1
            batch += 1
        return val + (1 - mask) * VERY_NEGATIVE_NUMBER


class SigmoidLoss(nn.Module):
    def __init__(self, hidden_size, word_emb_size):
        super(SigmoidLoss, self).__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size*2+word_emb_size*2, 1))
        utils.init_weight(self.weight)


    def forward(self, ctx_rep, men_rep, labels, types_emb):
        """
        args:
            ctx_rep: (B, hidden_size*2)
            men_rep: (B, word_emb)
            labels: (B, 1)
            types_emb: (B, word_emb)
        :return:
        """
        # batch_weight = torch.mm(types_emb, self.weight).unsqueeze(1)  # (B, 1, hidden_size*2+word_emb)
        # rep = torch.cat((ctx_rep, men_rep), 1).unsqueeze(2)  # (B, hidden_size*2+word_emb, 1)
        # logits = torch.bmm(batch_weight, rep).squeeze(2)  # (B, 1)
        # logits = torch.clamp(logits, max=16)
        # logits = torch.sigmoid(logits)  # (B, 1)
        logits = torch.mm(torch.cat((ctx_rep, men_rep, types_emb), 1), self.weight)  # (B, 1)
        logits = torch.clamp(logits, max=16)
        logits = torch.sigmoid(logits)  # (B, 1)
        loss = torch.sum(-labels*torch.log(logits)-(1-labels)*torch.log(1-logits)) / labels.size(0)
        return loss, logits


class NZSigmoidLoss(nn.Module):
    def __init__(self, hidden_size, word_emb_size):
        super(NZSigmoidLoss, self).__init__()
        require_type_lst = utils.get_ontoNotes_train_types()
        self.weight = nn.Parameter(torch.zeros(len(require_type_lst), hidden_size*2+word_emb_size))
        utils.init_weight(self.weight)

    def forward(self, ctx_rep, men_rep, labels):
        """
        args:
        :param ctx_rep: (B, hidden_size*2)
        :param men_rep: (B, word_emb)
        :param labels: （B, 89）
        :return:
        """
        if fg_config['att'] == 'label_att':
            # ctx_rep: (89, B, hidden_size * 2)
            types_len = self.weight.size(0)
            batch_size = men_rep.size(0)
            word_emb = men_rep.size(1)
            men_rep = men_rep.unsqueeze(0).expand(types_len, batch_size, word_emb)  # (89, B, word_emb)
            ctx_men = torch.cat((ctx_rep, men_rep), 2)  # (89, B, hidden_size * 2+word_emb)
            weight = self.weight.unsqueeze(2)  # (89, hidden_size * 2+word_emb, 1)
            logits = torch.bmm(ctx_men, weight).squeeze(2).transpose(0, 1)  # (B, 89)

        else:
            logits = torch.mm(torch.cat((ctx_rep, men_rep), 1), self.weight.transpose(0, 1))

        logits = torch.clamp(logits, max=16)
        logits = torch.sigmoid(logits)  # (B, 89)
        loss = torch.sum(-labels*torch.log(logits)-(1-labels)*torch.log(1-logits)) / labels.size(0)
        return loss, logits


class NZCtxAtt(nn.Module):
    def __init__(self, hidden_size, word_emb_size):
        super(NZCtxAtt, self).__init__()
        self.hidden_size = hidden_size
        self.word_emb_size = word_emb_size
        if fg_config['att'] == 'label_att':
            self.att_weight = nn.Parameter(torch.FloatTensor(hidden_size * 2, word_emb_size))
            utils.init_weight(self.att_weight)
        elif fg_config['att'] == 'orig_att':
            self.We = nn.Parameter(torch.FloatTensor(hidden_size*2, fg_config['Da']))
            utils.init_weight(self.We)
            self.Wa = nn.Parameter(torch.FloatTensor(fg_config['Da'], 1))
            utils.init_weight(self.Wa)
        elif fg_config['att'] == 'no':
            self.att_weight = nn.Parameter(torch.FloatTensor(hidden_size * 2, word_emb_size))
            utils.init_weight(self.att_weight)

        self.softmax = nn.Softmax()

    def forward(self, l_ctx_lstm, r_ctx_lstm, mentions_emb, l_ctx_lens, r_ctx_lens, men_lens, types_emb=None):
        """
        get mention representation and context representation

        args:
            l_ctx_lstm: (S, B, hidden_size*2)
            r_ctx_lstm: (S, B, hidden_size*2)
            mentions_emb: (B, S, word_emb)
            l_ctx_lens: list
            r_ctx_lens: list
            men_lens: list
            types_emb: (89, word_emb)
        :return:
        """
        l_ctx_lstm = l_ctx_lstm.transpose(0, 1).contiguous()  # (B, S, hidden_size*2)
        r_ctx_lstm = r_ctx_lstm.transpose(0, 1).contiguous()  # (B, S, hidden_size*2)


        if fg_config['att'] == 'label_att':
            ctx_rep = self.label_att(l_ctx_lstm, r_ctx_lstm, l_ctx_lens, r_ctx_lens, types_emb)  # (89, B, hidden_size*2)
        elif fg_config['att'] == 'no':
            ctx_rep = torch.cat((l_ctx_lstm[:, -1, :self.hidden_size], r_ctx_lstm[:, 0, self.hidden_size:]), 1)  # (B, hidden_size*2)
        elif fg_config['att'] == 'orig_att':
            ctx_rep = self.orig_att(l_ctx_lstm, r_ctx_lstm, l_ctx_lens, r_ctx_lens)  # (B, hidden_size*2)



        men_rep = torch.sum(mentions_emb, 1)  # (B, word_emb)
        men_len_mask = Variable(torch.ones(l_ctx_lstm.size(0), 1))
        if fg_config['USE_CUDA']:
            men_len_mask = men_len_mask.cuda(l_ctx_lstm.get_device())
        for b, l in enumerate(men_lens):
            men_len_mask[b, 0] = l
        men_len_mask = men_len_mask.expand_as(men_rep)
        men_rep = men_rep / men_len_mask

        # ctx_rep: (B, hidden_size*4), men_rep: (B, word_emb)
        return ctx_rep, men_rep

    def label_att(self, l_ctx_lstm, r_ctx_lstm, l_ctx_lens, r_ctx_lens, types_emb):
        """

        :param l_ctx_lstm:  (B, S, hidden_size*2)
        :param r_ctx_lstm:  (B, S, hidden_size*2)
        :param l_ctx_lens:  list
        :param r_ctx_lens:  list
        :param types_emb:  (89, word_emb)
        :return:
        """
        batch_size = l_ctx_lstm.size(0)
        S = l_ctx_lstm.size(1)
        hid2 = l_ctx_lstm.size(2)
        types_len = types_emb.size(0)
        word_emb = types_emb.size(1)


        tmp = torch.mm(l_ctx_lstm.view(-1, hid2).contiguous(), self.att_weight)  # (B*S, word_emb)
        tmp = torch.mm(tmp, types_emb.transpose(0, 1)).view(batch_size, S, types_len).transpose(1, 2).transpose(0, 1).contiguous()  # (89, B, S)
        l_weights = NZCtxAtt.exp_mask_3d(tmp, l_ctx_lens)  # (89, B, S)
        l_weights = l_weights.view(-1, S).contiguous()  # (89*B, S)


        tmp = torch.mm(r_ctx_lstm.view(-1, hid2).contiguous(), self.att_weight)  # (B*S, word_emb)
        tmp = torch.mm(tmp, types_emb.transpose(0, 1)).view(batch_size, S, types_len).transpose(1, 2).transpose(0, 1).contiguous()  # (89, B, S)
        r_weights = NZCtxAtt.exp_mask_3d(tmp, r_ctx_lens)  # (89, B, S)
        r_weights = r_weights.view(-1, S).contiguous()  # (89*B, S)


        ctx_weights = torch.cat((l_weights, r_weights), 1)  # (89*B, 2*S)
        ctx_weights = self.softmax(ctx_weights).view(types_len, batch_size, 2*S)  # (89, B, 2*S)
        l_weights = ctx_weights[:, :, :S]  # (89, B, S)
        r_weights = ctx_weights[:, :, S:]  # (89, B, S)



        # l_weights = self.softmax(l_weights).view(types_len, batch_size, S)  # (89, B, S)
        l_weights = l_weights.unsqueeze(3).expand(types_len, batch_size, S, hid2)  # (89, B, S, hidden_size*2)
        ex_l_ctx_lstm = l_ctx_lstm.unsqueeze(0).expand(types_len, batch_size, S, hid2)  # (89, B, S, hidden_size*2)
        l_ctx = torch.sum(l_weights * ex_l_ctx_lstm, 2)  # (89, B, hidden_size*2)


        # r_weights = self.softmax(r_weights).view(types_len, batch_size, S)  # (89, B, S)
        r_weights = r_weights.unsqueeze(3).expand(types_len, batch_size, S, hid2)  # (89, B, S, hidden_size*2)
        ex_r_ctx_lstm = r_ctx_lstm.unsqueeze(0).expand(types_len, batch_size, S, hid2)  # (89, B, S, hidden_size*2)
        r_ctx = torch.sum(r_weights * ex_r_ctx_lstm, 2)  # (89, B, hidden_size*2)

        ctx_rep = l_ctx + r_ctx  # (89, B, hidden_size*2)


        # ctx_rep: # (89, B, hidden_size*2)
        return ctx_rep

    def orig_att(self, l_ctx_lstm, r_ctx_lstm, l_ctx_lens, r_ctx_lens):
        """

        :param l_ctx_lstm:  (B, S, hidden_size*2)
        :param r_ctx_lstm:  (B, S, hidden_size*2)
        :param l_ctx_lens:  list
        :param r_ctx_lens:  list
        :return:
        """
        # We: (hidden_size*2, Da)  Wa: (Da, 1)

        batch_size = l_ctx_lstm.size(0)
        S = l_ctx_lstm.size(1)
        hid2 = l_ctx_lstm.size(2)


        tmp = torch.mm(l_ctx_lstm.view(-1, l_ctx_lstm.size(2)).contiguous(), self.We)  # (B*S, Da)
        l_weights = torch.mm(tmp, self.Wa).squeeze(1).view(batch_size, S)  # (B, S)
        l_weights = CtxAtt.exp_mask(l_weights, l_ctx_lens)  # (B, S)

        tmp = torch.mm(r_ctx_lstm.view(-1, r_ctx_lstm.size(2)).contiguous(), self.We)  # (B*S, Da)
        r_weights = torch.mm(tmp, self.Wa).squeeze(1).view(batch_size, S)  # (B, S)
        r_weights = CtxAtt.exp_mask(r_weights, r_ctx_lens)  # (B, S)


        ctx_weights = torch.cat((l_weights, r_weights), 1)  # (B, 2*S)
        ctx_weights = self.softmax(ctx_weights)  # (B, 2*S)
        l_weights = ctx_weights[:, :S]  # (B, S)
        r_weights = ctx_weights[:, S:]  # (B, S)


        l_weights = l_weights.unsqueeze(2).expand_as(l_ctx_lstm)  # (B, S, hidden_size*2)
        l_ctx = torch.sum(l_weights * l_ctx_lstm, 1)  # (B, hidden_size*2)
        r_weights = r_weights.unsqueeze(2).expand_as(r_ctx_lstm)  # (B, S, hidden_size*2)
        r_ctx = torch.sum(r_weights * r_ctx_lstm, 1)  # (B, hidden_size*2)

        ctx_rep = l_ctx + r_ctx  # (B, hidden_size*2)

        return ctx_rep




    @staticmethod
    def exp_mask(val, lens):
        VERY_BIG_NUMBER = 1e30
        VERY_SMALL_NUMBER = 1e-30
        VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
        VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
        mask = Variable(torch.zeros(val.size()))
        if fg_config['USE_CUDA']:
            mask = mask.cuda(val.get_device())
        batch = 0
        for i in lens:
            if i > 0:
                mask[batch, :i] = 1
            batch += 1
        return val + (1 - mask) * VERY_NEGATIVE_NUMBER

    @staticmethod
    def exp_mask_3d(val, lens):
        VERY_BIG_NUMBER = 1e30
        VERY_SMALL_NUMBER = 1e-30
        VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
        VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
        mask = Variable(torch.zeros(val.size()))
        if fg_config['USE_CUDA']:
            mask = mask.cuda(val.get_device())
        batch = 0
        for i in lens:
            if i > 0:
                mask[:, batch, :i] = 1
            batch += 1
        return val + (1 - mask) * VERY_NEGATIVE_NUMBER


    @staticmethod
    def average(val, lens):
        """
        args
        :param val: (B, S, d)
        :return:
        """
        rep = torch.sum(val, 1)  # (B, d)
        len_mask = Variable(torch.ones(val.size(0), 1))
        if fg_config['USE_CUDA']:
            len_mask = len_mask.cuda(val.get_device())
        for b, l in enumerate(lens):
            len_mask[b, 0] = l
        len_mask = len_mask.expand_as(rep)
        rep = rep / len_mask
        return rep

































