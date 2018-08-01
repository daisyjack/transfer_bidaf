from __future__ import print_function
import torch
from configurations import fg_config, to_np
from logger import Logger
import time
from entity_typing.et_module import EmbeddingLayer, CtxLSTM, NZCtxAtt
from bidaf import LoadEmbedding
from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
from numpy import linalg as LA
import sys
import os
from conll_data_trans import OntoNotesFGGetter, OntoNotesNZGetter
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



    def evaluate(self, label, pred, types_str):
        tmp = (pred > 0.5).astype(int)
        batch_num = 0
        for label_sample, pred_sample, this_types in zip(label, pred, types_str):
            pred_set = set([])
            truth_set = set([])
            tmp = (pred_sample > 0.5).astype(int)
            if numpy.sum(tmp) == 0:
                max_id = numpy.argwhere(pred_sample == numpy.max(pred_sample))
                pred_set.update(max_id.flatten().tolist())
                self._rec_num += len(pred_set)
                truth_set.update(numpy.argwhere(label_sample == numpy.max(label_sample)).flatten().tolist())
                self._hit_num += len(pred_set.intersection(truth_set))
            else:
                for t, p in zip(label_sample, tmp):
                    if p == 1:
                        self._rec_num += 1
                    if t == 1 and p == 1:
                        self._hit_num += 1
            self._lab_num += len(this_types)

    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print('label={}, rec={}, hit={}'.format(self._lab_num, self._rec_num, self._hit_num))
        print('p={},r={}, f={}'.format(p*100, r*100, f))
        return (f, p, r)




def evaluate_one(step, word_embedding_layer, type_embedding_layer, ctx_lstm, ctx_att, sigmoid_loss, this_batch):
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

    l_ctx_emb = word_embedding_layer(l_ctx)  # (B, S, word_emb)
    mentions_emb = word_embedding_layer(mentions)  # (B, S, word_emb)
    r_ctx_emb = word_embedding_layer(r_ctx)  # (B, S, word_emb)
    types_emb = type_embedding_layer(types)  # (B, word_emb)
    l_ctx_lstm, r_ctx_lstm = ctx_lstm(l_ctx_emb, r_ctx_emb, l_ctx_lens, r_ctx_lens)
    ctx_rep, men_rep = ctx_att(l_ctx_lstm, r_ctx_lstm, mentions_emb, l_ctx_lens, r_ctx_lens, men_lens, types_emb)
    _, logits = sigmoid_loss(ctx_rep, men_rep, labels)
    tmp = logits.cpu().data.numpy()  # (B, 89)
    # tmp = (tmp > 0.5).astype(int)
    l_tmp = this_batch['labels_tensor'].numpy()
    l_tmp = l_tmp.astype(int)
    return tmp, l_tmp

def get_type_lst(data):
    type_lst = None
    if data == 'onto':
        type_lst = utils.get_ontoNotes_train_types()
    elif data == 'wiki':
        type_lst = utils.get_wiki_types()
    elif data == 'bbn':
        type_lst = utils.get_bbn_types()
    return type_lst



def evaluate_all(my_arg, word_embedding_layer, type_embedding_layer, ctx_lstm, ctx_att, sigmoid_loss, pr=True):
    word_emb = LoadEmbedding('res/glove_840B_emb.txt')
    type_emb = LoadEmbedding('res/{}/zero_type_emb.txt'.format(fg_config['data']))
    print('finish loading embedding')
    batch_size = 100
    batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']), get_type_lst(fg_config['data']),
                                     batch_size, True)
    print('finish loading train data')
    # ctx_lstm = CtxLSTM(word_emb.get_emb_size())
    # embedding_layer = EmbeddingLayer(word_emb)
    # ctx_att = NZCtxAtt(fg_config['hidden_size'], word_emb.get_emb_size())
    # sigmoid_loss = NZSigmoidLoss(fg_config['hidden_size'], word_emb.get_emb_size())
    #
    # if fg_config['USE_CUDA']:
    #     embedding_layer.cuda(fg_config['cuda_num'])
    #     ctx_lstm.cuda(fg_config['cuda_num'])
    #     ctx_att.cuda(fg_config['cuda_num'])
    #     sigmoid_loss.cuda(fg_config['cuda_num'])
    # model_dir = 'nz_et_model' + str(my_arg)
    # embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    # ctx_lstm.load_state_dict(torch.load(model_dir+'/ctx_lstm.pkl'))
    # ctx_att.load_state_dict(torch.load(model_dir+'/ctx_att.pkl'))
    # sigmoid_loss.load_state_dict(torch.load(model_dir+'/sigmoid_loss.pkl'))
    word_embedding_layer.eval()
    type_embedding_layer.eval()
    ctx_lstm.eval()
    ctx_att.eval()
    sigmoid_loss.eval()
    ex_iterations = 0
    evaluator = BoundaryPerformance()
    for iteration, this_batch in enumerate(batch_getter):
        pred, label = evaluate_one(ex_iterations + iteration, word_embedding_layer, type_embedding_layer, ctx_lstm,
                                   ctx_att, sigmoid_loss, this_batch)

        evaluator.evaluate(label, pred, this_batch['types_str'])
        if (iteration+1)*batch_size % 100 == 0:
            print('{} sentences processed'.format((iteration+1)*batch_size))
            evaluator.get_performance()
    return evaluator.get_performance()


if __name__ == '__main__':
    fg_config['cuda_num'] = 0
    fg_config['batch_size'] = 64
    torch.cuda.set_device(fg_config['cuda_num'])
    evaluate_all(0)


