# coding: utf-8
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import utils

START_TAG = "<START>"
STOP_TAG = "<PADDING>"
VERY_BIG_NUMBER = 1e30
VERY_NEG_NUMBER = -1e30

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp(vec):
    """
    calculate log of exp sum

    args:
        vec (batch_size, to_target, from_target) : input tensor
    return:
        batch_size, hidden_dim
    """


    max_score, idx = torch.max(vec, -1, keepdim = True)  # ( B, to_target, 1)
    # max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    # max_score.expand_as(vec)
    # to_target = vec.size(1)

    return max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), -1))  # B * to_target


class CRF(nn.Module):

    # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    # tag_to_ix: config['BioOutTags']
    def __init__(self, config, tag_to_ix, hidden_dim):
        super(CRF, self).__init__()
        self.config = config
        # self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        # tag_to_ix = tag_to_ix.word2id
        self.tag_to_ix = tag_to_ix
        self.target_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
        #                     num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=False)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.target_size, self.target_size))
        # torch.nn.init.uniform(self.transitions, -config['weight_scale'], config['weight_scale'])
        self.transitions.data.zero_()



        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        if config['transition']:
            self.transitions.data[tag_to_ix[START_TAG], :] = VERY_NEG_NUMBER
            self.transitions.data[:, tag_to_ix[STOP_TAG]] = VERY_NEG_NUMBER


    # small_crf: feats: (seq_len, batch, target_size)  large_crf: feats: (seq_len, batch, target_size*target_size)
    def compute_scores(self, feats, lengths):
        seq_length = feats.size(0)
        batch_size = feats.size(1)
        if self.config['large_crf']:
            trans = self.transitions.view(1, 1, self.target_size, self.target_size).expand(seq_length, batch_size,
                                                                                           self.target_size,
                                                                                           self.target_size)

            mask = Variable(CRF.get_source_mask(batch_size, self.target_size, max(lengths), lengths))
            if self.config['USE_CUDA']:
                mask = mask.cuda(feats.get_device())
            feats = feats.view(seq_length, batch_size, self.target_size, self.target_size)
            scores = feats + trans * mask
        else:
            scores = feats.view(-1, self.target_size, 1)
            ins_num = scores.size(0)
            crf_scores = scores.expand(ins_num, self.target_size, self.target_size) + \
                         self.transitions.view(1, self.target_size, self.target_size).expand(ins_num, self.target_size,
                                                                                             self.target_size)
            # scores: (seq_len, batch, to_tagsize, from_tagsize) 下一个step加上的score
            scores = crf_scores.view(seq_length, batch_size, self.target_size, self.target_size)





        return scores

    @staticmethod
    def get_source_mask(batch, target_size, max_length, lengths):
        mask = torch.ones(batch, max_length, target_size, target_size)
        for i in range(batch):
            mask[i, :lengths[i], :, :] = 0
        return mask.transpose(0, 1)


    # small_crf: feats: (seq_len, batch, target_size)  large_crf: feats: (seq_len, batch, target_size*target_size)
    # target: (seq_len, bat_size)
    # mask: (seq_len, bat_size)
    def _forward_alg(self, feats, target, mask, lengths):
        seq_length = feats.size(0)
        batch_size = feats.size(1)
        # return 0.001*torch.mm(self.transitions, Variable(torch.randn(self.tagset_size, self.tagset_size)).cuda(0)).sum()


        # scores: (seq_len, batch, to_tagsize, from_tagsize) 下一个step加上的score
        scores = self.compute_scores(feats, lengths)


        
        start_tag = Variable(torch.LongTensor(1, batch_size).fill_(self.tag_to_ix[START_TAG]))
        end_tag = Variable(torch.LongTensor(1, batch_size).fill_(self.tag_to_ix[STOP_TAG]))
        if self.config['USE_CUDA']:
            start_tag = start_tag.cuda(feats.get_device())
            end_tag = end_tag.cuda(feats.get_device())

        from_tag = torch.cat([start_tag, target], 0)
        # to_tag: (seq_len+1, batch)
        to_tag = torch.cat([target, end_tag], 0)
        # scores_add_an_end: (seq_len+1, batch, to_tagsize, from_tagsize)
        scores_add_an_end = torch.cat([scores, self.transitions.view(1, 1, self.target_size, self.target_size)
                                      .expand(1, batch_size, self.target_size, self.target_size)], 0)
        # from_tag: (seq_len+1, batch, to_tagsize, 1)
        from_tag = from_tag.view(seq_length+1, batch_size, 1, 1)\
            .expand(seq_length + 1, batch_size, self.target_size, 1)
        # from_score: (seq_len+1, batch, to_tagsize)
        from_score = torch.gather(scores_add_an_end, -1, from_tag).squeeze(-1)
        # to_tag: (seq_len+1, batch, 1)
        to_tag = to_tag.unsqueeze(-1)
        # sen_scores: (seq_len+1, batch)
        sen_scores = torch.gather(from_score, -1, to_tag).squeeze(-1)
        # mask_add_an_end: (seq_len+1, bat_size)
        one_mask = Variable(torch.ByteTensor(1, batch_size).fill_(1))
        if self.config['USE_CUDA']:
            one_mask = one_mask.cuda(feats.get_device())
        mask_add_an_end = torch.cat([one_mask, mask], 0)
        masked_sen_scores = sen_scores.masked_select(mask_add_an_end).sum()


        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.target_size).fill_(VERY_NEG_NUMBER)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)
        if self.config['USE_CUDA']:
            forward_var = forward_var.cuda(feats.get_device())
        forward_var = forward_var.expand(batch_size, self.target_size)  # (batch, from_target)
        for idx in range(scores.size(0)):
            cur_values = scores[idx]
            # brod_forward_var: (batch, to_target, from_target)
            brod_forward_var = forward_var.unsqueeze(1).expand(batch_size, self.target_size, self.target_size)
            next_step_var = brod_forward_var + cur_values  # (batch_size, to_target, from_target)
            next_step_var = log_sum_exp(next_step_var)  # (batch, to_target)
            # (batch, to_target) 下一个循环变成from_target
            forward_var = utils.switch(forward_var.contiguous(), next_step_var.contiguous(),
                                       mask[idx].view(batch_size, 1).expand(batch_size, self.target_size)).view(batch_size, -1)
        # (B, from_target)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0).expand_as(forward_var)
        # terminal_var = util.switch(forward_var.contiguous(), terminal_var.contiguous(),
        #                           mask[idx].view(batch_size, 1).expand(batch_size, self.tagset_size)).view(batch_size,
        #                                                                                                    -1)

        alpha = log_sum_exp(terminal_var)  # (B, )
        all = alpha.sum()
        loss = (all - masked_sen_scores) / batch_size
        return loss



    # def _get_lstm_features(self, lstm_out):
    #     lstm_feats = self.hidden2tag(lstm_out)
    #     return lstm_feats

    # feats: (seq_len, batch, target_size)
    # mask: (seq_len, bat_size)
    def _viterbi_decode(self, feats, mask, lengths):
        seq_length = feats.size(0)
        batch_size = feats.size(1)


        # scores: (seq_len, batch, to_tagsize, from_tagsize) 下一个step加上的score
        scores = self.compute_scores(feats, lengths)


        scores_add_an_end = torch.cat([scores, self.transitions.view(1, 1, self.target_size, self.target_size)
                                      .expand(1, batch_size, self.target_size, self.target_size)], 0)
        one_mask = Variable(torch.ByteTensor(1, batch_size).fill_(1))
        if self.config['USE_CUDA']:
            one_mask = one_mask.cuda(feats.get_device())
        # mask_add_an_end: (seq_len+1, bat_size)
        mask_add_an_end = torch.cat([one_mask, mask], 0)


        # Do the forward algorithm to compute the partition function
        init_alphas = torch.FloatTensor(1, self.target_size).fill_(VERY_NEG_NUMBER)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = Variable(init_alphas)
        if self.config['USE_CUDA']:
            forward_var = forward_var.cuda(feats.get_device())
        forward_var = forward_var.expand(batch_size, self.target_size)  # (batch, from_target)
        # # forward_var: (batch_size, tagset_size, tagset_size)
        # forward_var = forward_var.unsqueeze(1).expand(self.batch_size, self.tagset_size, self.tagset_size)  # (bath, to_target, from_target)
        # (seq_len+1)*(batch, to_target)
        back_points = []

        for idx, cur_values in enumerate(scores_add_an_end):
            # forward_var: (bath, to_target, from_target)
            forward_var = forward_var.unsqueeze(1).expand(batch_size, self.target_size, self.target_size)
            next_step_var = forward_var + cur_values  # (batch_size, to_target, from_target)
            # best_score: (batch, to_target) best_id: (batch, to_target)
            best_score, best_id = torch.max(next_step_var, 2)
            oppo_mask = mask_add_an_end[idx].view(batch_size, 1).expand(batch_size, self.target_size) == 0
            best_id.masked_fill_(oppo_mask, self.tag_to_ix[STOP_TAG])
            back_points.append(best_id)
            # (batch, to_target) 下一个循环变成from_target
            forward_var = best_score


        decode_idx = Variable(torch.LongTensor(seq_length, batch_size))
        if self.config['USE_CUDA']:
            decode_idx = decode_idx.cuda(feats.get_device())
        # pointer: (batch,)
        pointer = back_points[-1][:, self.tag_to_ix[STOP_TAG]]
        decode_idx[-1, :] = pointer
        for idx in range(len(back_points)-2, 0, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.unsqueeze(1)).squeeze(1)
            decode_idx[idx-1, :] = pointer
        # lst_decode: batch_size
        lst_decode = [self.remove_endtag(one_batch) for one_batch in decode_idx.transpose(0,1)]

        return lst_decode

    # tag_lst(LongTensor): (seq_length, )
    def remove_endtag(self, tag_var):
        tag_list = tag_var.cpu().data.numpy().tolist()
        new_tag = []
        for tag in tag_list:
            if tag == self.tag_to_ix[STOP_TAG]:
                break
            else:
                new_tag.append(tag)

        return new_tag

    # small_crf: sentence: (seq_len, batch, target_size)  large_crf: sentence: (seq_len, batch, target_size*target_size)
    # target: (seq_len, bat_size)
    # mask: (seq_len, bat_size)
    def neg_log_likelihood(self, sentence, tags, mask, lengths):
        # feats = self._get_lstm_features(sentence)  # (seq_len, batch, hidden_size*directions)
        feats = sentence

        loss = self._forward_alg(feats, tags, mask, lengths)
        return loss

    # small_crf: feats: (seq_len, batch, target_size)  large_crf: feats: (seq_len, batch, target_size*target_size)
    # mask: (seq_len, bat_size)
    def forward(self, sentence, mask, lengths):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # lstm_feats = self._get_lstm_features(sentence)
        lstm_feats = sentence

        # Find the best path, given the features.
        tag_seq = self._viterbi_decode(lstm_feats, mask, lengths)
        return tag_seq


if __name__ == '__main__':
    print(log_sum_exp(torch.FloatTensor([1,1,1])))


