from __future__ import print_function
import torch
from configurations import fg_config, to_np, Vocab
from logger import Logger
import time
from entity_typing.et_module import EmbeddingLayer, CtxLSTM, NZSigmoidLoss, NZCtxAtt, WARPLoss
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



def get_short2full_map(require_type_lst):
    short2full = {}
    for full in require_type_lst:
        short = full.split('/')[-1]
        short2full[short] = full
    return short2full


def refine(labels, short2full, maxDepth=3):
    keep = [""] * maxDepth
    # short2full = get_short2full_map(utils.get_ontoNotes_train_types())
    for l in labels:
        path = l.split('/')[1:]
        path = [short2full[k] for k in path]
        for i in range(len(path)):
            if keep[i] == "":
                keep[i] = path[i]
            elif keep[i] != path[i]:
                break

    return [l for l in keep if l != ""]


class BoundaryPerformance:
    def __init__(self):
        self.reset()

    def reset(self):
        self._hit_num = 0
        self._rec_num = 0
        self._lab_num = 0
        self._unmatch = 0



    def evaluate(self, label, pred, type_lst, short2full):

        for pred_sample, label_sample in zip(pred, label):
            # label_sample = numpy.where(label_sample)[0].tolist()
            pred_sample = pred_sample.tolist()[::-1]
            # label_sample = [type_lst[l] for l in label_sample]
            pred_sample = [type_lst[p] for p in pred_sample]
            pred_sample = refine(pred_sample, short2full)
            self._rec_num += len(pred_sample)
            self._lab_num += len(label_sample)
            pred_set = set(pred_sample)
            label_set = set(label_sample)
            self._hit_num += len(pred_set.intersection(label_set))


    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print('label={}, rec={}, hit={}'.format(self._lab_num, self._rec_num, self._hit_num))
        print('p={},r={}, f={}'.format(p*100, r*100, f))
        return (f, p, r)




def evaluate_one(step, word_embedding_layer, type_embedding_layer, ctx_lstm, ctx_att, warp_loss, this_batch):
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
    logits = warp_loss.get_scores(ctx_rep, men_rep, types_emb)
    tmp = logits.cpu().data.numpy()  # (B, 89)
    # tmp = (tmp > 0.5).astype(int)
    tmp = numpy.argsort(tmp, axis=1)[:, -fg_config['topk']:]  # (B, topk)
    l_tmp = this_batch['labels_tensor'].numpy()
    l_tmp = l_tmp.astype(int)
    return tmp, l_tmp


def get_type_lst(depth, data):
    type_lst = None
    if data == 'onto':
        type_lst = utils.get_ontoNotes_train_types(depth)
    elif data == 'wiki':
        type_lst = utils.get_wiki_types(depth)
    elif data == 'bbn':
        type_lst = utils.get_bbn_types(depth)
    return type_lst


def evaluate_all(my_arg, word_embedding_layer, type_embedding_layer, ctx_lstm, ctx_att, warp_loss, pr=True):
    word_emb = LoadEmbedding('res/glove_840B_emb.txt')
    type_emb = LoadEmbedding('res/{}/zero_type_emb.txt'.format(fg_config['data']))
    print('finish loading embedding')
    batch_size = 100
    depth = None
    if fg_config['zero_shot']:
        depth = 2
    elif fg_config['no_zero'] == 'all':
        depth = None
    elif fg_config['no_zero'] == 'one':
        depth = 1

    type_lst = get_type_lst(depth, fg_config['data'])
    batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']),
                                     type_lst, batch_size, True, depth)

    # if fg_config['zero_shot']:
    #     batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(2),
    #                                  batch_size, True, 2)
    # elif fg_config['no_zero'] == 'all':
    #     batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(),
    #                                      batch_size, True, None)
    # elif fg_config['no_zero'] == 'one':
    #     batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(1),
    #                                      batch_size, True, 1)
    print('finish loading train data')
    # ctx_lstm = CtxLSTM(word_emb.get_emb_size())
    # word_embedding_layer = EmbeddingLayer(word_emb)
    # type_embedding_layer = EmbeddingLayer(type_emb)
    # ctx_att = NZCtxAtt(fg_config['hidden_size'], word_emb.get_emb_size())
    # warp_loss = WARPLoss(fg_config['hidden_size'], word_emb.get_emb_size())
    #
    # if fg_config['USE_CUDA']:
    #     word_embedding_layer.cuda(fg_config['cuda_num'])
    #     type_embedding_layer.cuda(fg_config['cuda_num'])
    #     ctx_lstm.cuda(fg_config['cuda_num'])
    #     ctx_att.cuda(fg_config['cuda_num'])
    #     warp_loss.cuda(fg_config['cuda_num'])
    # model_dir = 'zero_et_model' + str(my_arg)
    # word_embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    # type_embedding_layer.load_state_dict(torch.load(model_dir+'/type_embedding_layer.pkl'))
    # ctx_lstm.load_state_dict(torch.load(model_dir+'/ctx_lstm.pkl'))
    # ctx_att.load_state_dict(torch.load(model_dir+'/ctx_att.pkl'))
    # warp_loss.load_state_dict(torch.load(model_dir+'/sigmoid_loss.pkl'))
    word_embedding_layer.eval()
    type_embedding_layer.eval()
    ctx_lstm.eval()
    ctx_att.eval()
    warp_loss.eval()
    ex_iterations = 0
    evaluator = BoundaryPerformance()
    # if fg_config['zero_shot']:
    #     type_lst = utils.get_ontoNotes_train_types(2)
    # else:
    #     type_lst = utils.get_ontoNotes_train_types()
    short2full = None
    if fg_config['data'] == 'onto':
        short2full = get_short2full_map(utils.get_ontoNotes_train_types())
    elif fg_config['data'] == 'wiki':
        short2full = get_short2full_map(utils.get_wiki_types())
        patch = utils.wiki_short2full_patch()
        short2full.update(patch)
    elif fg_config['data'] == 'bbn':
        short2full = get_short2full_map(utils.get_bbn_types())

    for iteration, this_batch in enumerate(batch_getter):
        pred, label = evaluate_one(ex_iterations + iteration, word_embedding_layer, type_embedding_layer,
                                   ctx_lstm, ctx_att, warp_loss, this_batch)

        # evaluator.evaluate(label, pred, type_lst, short2full)
        evaluator.evaluate(this_batch['types_str'], pred, type_lst, short2full)
        if (iteration+1)*batch_size % 100 == 0:
            print('{} sentences processed'.format((iteration+1)*batch_size))
            evaluator.get_performance()
    return evaluator.get_performance()


def get_down_type_lst(depth, data):
    type_lst = []
    if data == 'onto':
        for i in range(1, depth+1):
            type_lst.extend(utils.get_ontoNotes_train_types(i))
    elif data == 'wiki':
        for i in range(1, depth + 1):
            type_lst.extend(utils.get_wiki_types(i))
    elif data == 'bbn':
        for i in range(1, depth + 1):
            type_lst.extend(utils.get_bbn_types(i))
    return type_lst



def evaluate_free(my_arg, pr=True):
    word_emb = LoadEmbedding('res/glove_840B_emb.txt')
    type_emb = LoadEmbedding('res/{}/zero_type_emb.txt'.format(fg_config['data']))
    print('finish loading embedding')
    batch_size = 100
    depth = None
    if fg_config['zero_shot']:
        depth = 2
    elif fg_config['no_zero'] == 'all':
        depth = None
    elif fg_config['no_zero'] == 'one':
        depth = 1

    type_lst = get_down_type_lst(depth, fg_config['data'])
    batch_getter = OntoNotesNZGetter('data/{}/test.json'.format(fg_config['data']),
                                     type_lst, batch_size, True, depth)
    print('finish loading train data')
    ctx_lstm = CtxLSTM(word_emb.get_emb_size())
    word_embedding_layer = EmbeddingLayer(word_emb)
    type_embedding_layer = EmbeddingLayer(type_emb)
    ctx_att = NZCtxAtt(fg_config['hidden_size'], word_emb.get_emb_size())
    warp_loss = WARPLoss(fg_config['hidden_size'], word_emb.get_emb_size())

    if fg_config['USE_CUDA']:
        word_embedding_layer.cuda(fg_config['cuda_num'])
        type_embedding_layer.cuda(fg_config['cuda_num'])
        ctx_lstm.cuda(fg_config['cuda_num'])
        ctx_att.cuda(fg_config['cuda_num'])
        warp_loss.cuda(fg_config['cuda_num'])
    model_dir = '{}/et_model{}'.format(fg_config['data'], str(my_arg))
    word_embedding_layer.load_state_dict(torch.load(model_dir+'/early_embedding_layer.pkl'))
    type_embedding_layer.load_state_dict(torch.load(model_dir+'/early_type_embedding_layer.pkl'))
    ctx_lstm.load_state_dict(torch.load(model_dir+'/early_ctx_lstm.pkl'))
    ctx_att.load_state_dict(torch.load(model_dir+'/early_ctx_att.pkl'))
    warp_loss.load_state_dict(torch.load(model_dir+'/early_sigmoid_loss.pkl'))
    word_embedding_layer.eval()
    type_embedding_layer.eval()
    ctx_lstm.eval()
    ctx_att.eval()
    warp_loss.eval()
    ex_iterations = 0
    evaluator = BoundaryPerformance()
    short2full = None
    if fg_config['data'] == 'onto':
        short2full = get_short2full_map(utils.get_ontoNotes_train_types())
    elif fg_config['data'] == 'wiki':
        short2full = get_short2full_map(utils.get_wiki_types())
        patch = utils.wiki_short2full_patch()
        short2full.update(patch)
    elif fg_config['data'] == 'bbn':
        short2full = get_short2full_map(utils.get_bbn_types())
    for iteration, this_batch in enumerate(batch_getter):
        pred, label = evaluate_one(ex_iterations + iteration, word_embedding_layer, type_embedding_layer,
                                   ctx_lstm, ctx_att, warp_loss, this_batch)

        # evaluator.evaluate(label, pred, type_lst, short2full)
        evaluator.evaluate(this_batch['types_str'], pred, type_lst, short2full)
        if (iteration+1)*batch_size % 100 == 0:
            print('{} sentences processed'.format((iteration+1)*batch_size))
            evaluator.get_performance()
    return evaluator.get_performance()


if __name__ == '__main__':
    fg_config['cuda_num'] = 0
    fg_config['batch_size'] = 64
    fg_config['att'] = 'label_att'
    fg_config['zero_shot'] = True
    fg_config['no_zero'] = 'all'
    fg_config['topk'] = 3
    fg_config['data'] = 'bbn'
    fg_config['type_id'] = Vocab('res/{}/zero_type_voc.txt'.format(fg_config['data']), unk_id=fg_config['UNK_token'],
                                 pad_id=fg_config['PAD_token'])
    torch.cuda.set_device(fg_config['cuda_num'])
    evaluate_free(0)


