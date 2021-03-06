from __future__ import print_function
import torch
from configurations import fg_config, to_np
from logger import Logger
import time
from entity_typing.et_module import EmbeddingLayer, CtxLSTM, CtxAtt, SigmoidLoss
from bidaf import LoadEmbedding
from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
from numpy import linalg as LA
import sys
import os
from conll_data_trans import OntoNotesFGGetter
import argparse
from batch_getter import get_source_mask, get_target_mask
from torch import nn
import random
import utils


class BoundaryPerformance:
    def __init__(self):
        self.reset()

    def reset(self):
        self._hit_num = 0
        self._rec_num = 0
        self._lab_num = 0
        self._unmatch = 0



    def evaluate(self, label, pred):
        for t, p in zip(label, pred):
            if t == 1:
                self._lab_num += 1
            if p == 1:
                self._rec_num += 1
            if t == 1 and p == 1:
                self._hit_num += 1

    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print('label={}, rec={}, hit={}'.format(self._lab_num, self._rec_num, self._hit_num))
        print('p={},r={}, f={}'.format(p*100, r*100, f))
        return (f, p, r)




def evaluate_one(step, embedding_layer, ctx_lstm, ctx_att, sigmoid_loss, this_batch):
    l_ctx = Variable(this_batch['l_ctx_tensor'], volatile=True)
    mentions = Variable(this_batch['mentions_tensor'], volatile=True)
    r_ctx = Variable(this_batch['r_ctx_tensor'], volatile=True)
    types = Variable(this_batch['types_tensor'], volatile=True)
    labels = Variable(this_batch['labels_tensor'], volatile=True)
    if fg_config['USE_CUDA']:
        l_ctx = l_ctx.cuda(fg_config['cuda_num'])
        mentions = mentions.cuda(fg_config['cuda_num'])
        r_ctx = r_ctx.cuda(fg_config['cuda_num'])
        types = types.cuda(fg_config['cuda_num'])
        labels = labels.cuda(fg_config['cuda_num'])
    l_ctx_lens = this_batch['l_ctx_lens']
    r_ctx_lens = this_batch['r_ctx_lens']
    men_lens = this_batch['men_lens']

    l_ctx_emb = embedding_layer(l_ctx)  # (B, S, word_emb)
    mentions_emb = embedding_layer(mentions)  # (B, S, word_emb)
    r_ctx_emb = embedding_layer(r_ctx)  # (B, S, word_emb)
    types_emb = embedding_layer(types)  # (B, word_emb)
    l_ctx_lstm, r_ctx_lstm = ctx_lstm(l_ctx_emb, r_ctx_emb, l_ctx_lens, r_ctx_lens)
    ctx_rep, men_rep = ctx_att(l_ctx_lstm, r_ctx_lstm, types_emb, mentions_emb, l_ctx_lens, r_ctx_lens, men_lens)
    _, logits = sigmoid_loss(ctx_rep, men_rep, labels, types_emb)
    tmp = logits.cpu().data.squeeze(1).numpy()  # (B, )
    tmp = (tmp > 0.5).astype(int)
    l_tmp = this_batch['labels_tensor'].squeeze(1).numpy()
    l_tmp = l_tmp.astype(int)
    return tmp.tolist(), l_tmp.tolist()





def evaluate_all(my_arg, pr=True):
    emb = LoadEmbedding('res/onto_emb.txt')
    print('finish loading embedding')
    batch_size = 1000
    batch_getter = OntoNotesFGGetter('data/OntoNotes/test.json', utils.get_ontoNotes_train_types(),
                                     batch_size, True)
    print('finish loading train data')
    ctx_lstm = CtxLSTM(emb.get_emb_size())
    embedding_layer = EmbeddingLayer(emb)
    ctx_att = CtxAtt(fg_config['hidden_size'], emb.get_emb_size())
    sigmoid_loss = SigmoidLoss(fg_config['hidden_size'], emb.get_emb_size())

    if fg_config['USE_CUDA']:
        embedding_layer.cuda(fg_config['cuda_num'])
        ctx_lstm.cuda(fg_config['cuda_num'])
        ctx_att.cuda(fg_config['cuda_num'])
        sigmoid_loss.cuda(fg_config['cuda_num'])
    model_dir = 'et_model' + str(my_arg)
    embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    ctx_lstm.load_state_dict(torch.load(model_dir+'/ctx_lstm.pkl'))
    ctx_att.load_state_dict(torch.load(model_dir+'/ctx_att.pkl'))
    sigmoid_loss.load_state_dict(torch.load(model_dir+'/sigmoid_loss.pkl'))
    embedding_layer.eval()
    ctx_lstm.eval()
    ctx_att.eval()
    sigmoid_loss.eval()
    ex_iterations = 0
    evaluator = BoundaryPerformance()
    for iteration, this_batch in enumerate(batch_getter):
        pred, label = evaluate_one(ex_iterations + iteration, embedding_layer, ctx_lstm, ctx_att, sigmoid_loss, this_batch)

        evaluator.evaluate(label, pred)
        if (iteration+1)*batch_size % 100 == 0:
            print('{} sentences processed'.format((iteration+1)*batch_size))
            evaluator.get_performance()
    return evaluator.get_performance()


if __name__ == '__main__':
    fg_config['cuda_num'] = 0
    fg_config['batch_size'] = 64
    torch.cuda.set_device(fg_config['cuda_num'])
    evaluate_all(0)


