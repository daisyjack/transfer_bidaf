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
from eval_zero_shot_et import evaluate_all
from numpy import linalg as LA
import sys
import os
from conll_data_trans import OntoNotesFGGetter, OntoNotesNZGetter
import argparse
from batch_getter import get_source_mask, get_target_mask
from torch import nn
import random
import utils
import sys
sys.path.append('/home/ryk/pycharm-debug-py3k.egg')
# from itertools import ifilter
import pydevd

torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)





def train_iteration(logger, step, word_embedding_layer, type_embedding_layer, ctx_lstm, ctx_att, warp_loss,
                    ctx_lstm_opt, ctx_att_opt, sig_opt, this_batch):
    # if step == 398:
    #     pydevd.settrace('10.214.129.230', port=31235, stdoutToServer=True, stderrToServer=True)

    l_ctx = Variable(this_batch['l_ctx_tensor'])
    mentions = Variable(this_batch['mentions_tensor'])
    r_ctx = Variable(this_batch['r_ctx_tensor'])
    types = Variable(this_batch['types_tensor'])
    labels = Variable(this_batch['labels_tensor'])
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
    types_emb = type_embedding_layer(types)  # (89, word_emb)
    l_ctx_lstm, r_ctx_lstm = ctx_lstm(l_ctx_emb, r_ctx_emb, l_ctx_lens, r_ctx_lens)
    ctx_rep, men_rep = ctx_att(l_ctx_lstm, r_ctx_lstm, mentions_emb, l_ctx_lens, r_ctx_lens, men_lens, types_emb)

    loss = warp_loss(ctx_rep, men_rep, labels, types_emb)
    if step > 100:
        print('loss: ', loss.data[0])
        logger.scalar_summary('loss', loss.data[0], step)


    loss.backward()

    ctx_lstm_before_step = [(tag, to_np(value)) for tag, value in ctx_lstm.named_parameters()]
    ctx_att_before_step = [(tag, to_np(value)) for tag, value in ctx_att.named_parameters()]
    sig_before_step = [(tag, to_np(value)) for tag, value in warp_loss.named_parameters()]




    # clip_grad_norm(embedding_layer.parameters(), fg_config['clip_norm'])
    clip_grad_norm(ctx_lstm.parameters(), fg_config['clip_norm'])
    clip_grad_norm(ctx_att.parameters(), fg_config['clip_norm'])
    clip_grad_norm(warp_loss.parameters(), fg_config['clip_norm'])
    # for tag, value in embedding_layer.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    # for tag, value in att_layer.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    # for tag, value in model_out_layer.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)

    # for tag, value in ner_out_layer.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    #
    # for tag, value in crf.named_parameters():
    #     tag = tag.replace('.', '/')
    #     if value is not None and value.grad is not None:
    #         logger.histo_summary(tag, to_np(value), step)
    #         logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    ctx_lstm_opt.step()
    ctx_att_opt.step()
    sig_opt.step()

    grad_ratio_lst = []

    ctx_lstm_after_step = [(tag, to_np(value)) for tag, value in ctx_lstm.named_parameters()]
    grad_ratio_lst.append((ctx_lstm_before_step, ctx_lstm_after_step))
    ctx_att_after_step = [(tag, to_np(value)) for tag, value in ctx_att.named_parameters()]
    grad_ratio_lst.append((ctx_att_before_step, ctx_att_after_step))
    sig_after_step = [(tag, to_np(value)) for tag, value in warp_loss.named_parameters()]
    grad_ratio_lst.append((sig_before_step, sig_after_step))
    # h_after_step = [(tag, to_np(value)) for tag, value in ner_hw_layer.named_parameters()]
    # grad_ratio_lst.append((h_before_step, h_after_step))
    # n_after_step = [(tag, to_np(value)) for tag, value in ner_out_layer.named_parameters()]
    # grad_ratio_lst.append((n_before_step, n_after_step))
    # c_after_step = [(tag, to_np(value)) for tag, value in crf.named_parameters()]
    # grad_ratio_lst.append((c_before_step, c_after_step))
    # q_after_step = [(tag, to_np(value)) for tag, value in q_emb_layer.named_parameters()]
    #
    utils.log_grad_ratio(logger, step, grad_ratio_lst)


def get_type_lst(depth, data):
    type_lst = None
    if data == 'onto':
        type_lst = utils.get_ontoNotes_train_types(depth)
    elif data == 'wiki':
        type_lst = utils.get_wiki_types(depth)
    elif data == 'bbn':
        type_lst = utils.get_bbn_types(depth)
    return type_lst



def main(my_arg):
    log_dir = '{}/zero_et_logs{}'.format(fg_config['data'], my_arg)
    logger = Logger(log_dir)
    # emb = LoadEmbedding('res/glove_840B_emb.txt')
    word_emb = utils.load_embedding('res/glove.840B.300d.dat')

    if fg_config['rand_emb']:
        type_emb = LoadEmbedding('res/{}/zero_type_emb.rand'.format(fg_config['data']))
    else:
        type_emb = LoadEmbedding('res/{}/zero_type_emb.txt'.format(fg_config['data']))
    print('finish loading embedding')
    depth = None
    if fg_config['zero_shot']:
        depth = 1
    elif fg_config['no_zero'] == 'all':
        depth = None
    elif fg_config['no_zero'] == 'one':
        depth = 1

    batch_getter = OntoNotesNZGetter('data/{}/train.json'.format(fg_config['data']), get_type_lst(depth, fg_config['data']),
                                     fg_config['batch_size'], True, depth)





    # if fg_config['zero_shot']:
    #     batch_getter = OntoNotesNZGetter('data/{}/train.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(1),
    #                                      fg_config['batch_size'], True, 1)
    # elif fg_config['no_zero'] == 'all':
    #     batch_getter = OntoNotesNZGetter('data/{}/train.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(),
    #                                      fg_config['batch_size'], True, None)
    # elif fg_config['no_zero'] == 'one':
    #     batch_getter = OntoNotesNZGetter('data/{}/train.json'.format(fg_config['data']), utils.get_ontoNotes_train_types(1),
    #                                      fg_config['batch_size'], True, 1)
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

    ctx_lstm_opt = torch.optim.Adam(ctx_lstm.parameters())
    ctx_att_opt = torch.optim.Adam(ctx_att.parameters())
    sig_opt = torch.optim.Adam(warp_loss.parameters())

    log_file = open('{}/log_file'.format(log_dir), 'w')
    f_max = 0
    low_epoch = 0
    ex_iterations = 0
    model_dir = '{}/et_model{}'.format(fg_config['data'], str(my_arg))
    time0 = time.time()
    for epoch in range(fg_config['max_epoch']):
        word_embedding_layer.train()
        type_embedding_layer.train()
        ctx_lstm.train()
        ctx_att.train()
        warp_loss.train()
        # f, p, r = evaluate_all(my_arg, False)
        for iteration, this_batch in enumerate(batch_getter):
            # if (ex_iterations + iteration) % 100 == 0:
            #     print('epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration))
            print('epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration))

            train_iteration(logger, ex_iterations + iteration, word_embedding_layer, type_embedding_layer, ctx_lstm,
                            ctx_att, warp_loss, ctx_lstm_opt, ctx_att_opt, sig_opt, this_batch)
            # if (ex_iterations + iteration) % 100 == 0:
            #     time1 = time.time()
            #     print('this iteration time: ', time1 - time0, '\n')
            #     time0 = time1
            time1 = time.time()
            print('this iteration time: ', time1 - time0, '\n')
            time0 = time1
            if (ex_iterations + iteration) % fg_config['save_freq'] == 0:
                torch.save(word_embedding_layer.state_dict(), model_dir+'/embedding_layer.pkl')
                torch.save(type_embedding_layer.state_dict(), model_dir + '/type_embedding_layer.pkl')
                torch.save(ctx_lstm.state_dict(), model_dir+'/ctx_lstm.pkl')
                torch.save(ctx_att.state_dict(), model_dir+'/ctx_att.pkl')
                torch.save(warp_loss.state_dict(), model_dir+'/sigmoid_loss.pkl')


        ex_iterations += iteration + 1
        batch_getter.reset()
        fg_config['use_dropout'] = False
        f, p, r = evaluate_all(my_arg, word_embedding_layer, type_embedding_layer,
                                   ctx_lstm, ctx_att, warp_loss, False)
        fg_config['use_dropout'] = True
        log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(epoch, f, p, r))
        log_file.flush()
        if f >= f_max:
            f_max = f
            low_epoch = 0
            # os.system('cp {}/embedding_layer.pkl {}/early_embedding_layer.pkl'.format(model_dir, model_dir))
            # os.system('cp {}/type_embedding_layer.pkl {}/early_type_embedding_layer.pkl'.format(model_dir, model_dir))
            # os.system('cp {}/ctx_lstm.pkl {}/early_ctx_lstm.pkl'.format(model_dir, model_dir))
            # os.system('cp {}/ctx_att.pkl {}/early_ctx_att.pkl'.format(model_dir, model_dir))
            # os.system('cp {}/sigmoid_loss.pkl {}/early_sigmoid_loss.pkl'.format(model_dir, model_dir))
            torch.save(word_embedding_layer.state_dict(), model_dir + '/early_embedding_layer.pkl')
            torch.save(type_embedding_layer.state_dict(), model_dir + '/early_type_embedding_layer.pkl')
            torch.save(ctx_lstm.state_dict(), model_dir + '/early_ctx_lstm.pkl')
            torch.save(ctx_att.state_dict(), model_dir + '/early_ctx_att.pkl')
            torch.save(warp_loss.state_dict(), model_dir + '/early_sigmoid_loss.pkl')

        else:
            low_epoch += 1
            log_file.write('low' + str(low_epoch) + '\n')
            log_file.flush()
        if low_epoch >= fg_config['early_stop']:
            break
    log_file.close()

if __name__ == '__main__':
    # print get_question(3)
    # my_arg = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='dataset')
    parser.add_argument('--my_arg', help='mode', type=int)
    parser.add_argument('--batch', help='batch', type=int)
    parser.add_argument('--cuda_num', help='cuda_num', type=int)
    parser.add_argument('--zero_shot', action='store_true', help='whether use zero-shot')
    parser.add_argument('--topk', help='topk while inference', type=int)
    parser.add_argument('--rand_emb', action='store_true', help='use random type embedding')
    parser.add_argument('--no_zero', default='all', help='all for all and one for train with level 1, test with level 1')
    parser.add_argument('--att', default='label_att')
    args = parser.parse_args()
    my_arg = args.my_arg
    batch_size = args.batch
    cuda_num = args.cuda_num
    fg_config['cuda_num'] = cuda_num
    fg_config['data'] = args.data
    fg_config['type_id'] = Vocab('res/{}/zero_type_voc.txt'.format(fg_config['data']), unk_id=fg_config['UNK_token'],
                                 pad_id=fg_config['PAD_token'])
    fg_config['batch_size'] = batch_size
    fg_config['att'] = args.att
    fg_config['zero_shot'] = args.zero_shot
    fg_config['topk'] = args.topk
    fg_config['rand_emb'] = args.rand_emb
    fg_config['no_zero'] = args.no_zero
    torch.cuda.set_device(fg_config['cuda_num'])
    print(fg_config['data'])
    main(my_arg)
