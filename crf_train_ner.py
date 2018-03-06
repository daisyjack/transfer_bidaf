from __future__ import print_function
import torch
from configurations import config, to_np
from logger import Logger
from batch_getter import BatchGetter, get_question, MergeBatchGetter
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingLayer, QEmbeddingLayer, NerOutLayer, NerHighway, QLabel
from crf import CRF
from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
from crf_eval_ner import evaluate_all
from numpy import linalg as LA
import sys
import os
from conll_data_trans import ConllBatchGetter, TrainConllBatchGetter, TrainOntoNotesGetter, OntoNotesGetter
import argparse
from batch_getter import get_source_mask, get_target_mask
from torch import nn
import random
import utils
# from itertools import ifilter
# import pydevd
#
# pydevd.settrace('10.214.129.230', port=31235, stdoutToServer=True, stderrToServer=True)
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)





def train_iteration(logger, step, embedding_layer, q_word_embedding, q_emb_layer, att_layer, model_layer, ner_hw_layer, ner_out_layer, crf,
                    emb_opt, q_emb_opt, att_opt, model_opt, ner_hw_opt, ner_out_opt, crf_opt, this_batch):
    if not config['freeze']:
        emb_opt.zero_grad()
        att_opt.zero_grad()
        model_opt.zero_grad()
    if config['question_alone']:
        q_emb_opt.zero_grad()
    ner_out_opt.zero_grad()
    crf_opt.zero_grad()
    ner_hw_opt.zero_grad()

    d = config['hidden_size']
    this_batch_num = len(this_batch[2])

    question = Variable(this_batch[4])
    question_lengths = this_batch[5]
    context = Variable(this_batch[0])  # (batch, T, 51)
    context_lengths = this_batch[3]  # list
    target = Variable(this_batch[1])  # (batch, T)
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d))
    model_out_h_0 = Variable(torch.zeros(2*model_layer.num_layers, this_batch_num, d))
    con_lens_var = Variable(torch.LongTensor(context_lengths))

    if config['USE_CUDA']:
        question = question.cuda(config['cuda_num'])
        context = context.cuda(config['cuda_num'])
        target = target.cuda(config['cuda_num'])
        emb_h_0 = emb_h_0.cuda(config['cuda_num'])
        model_out_h_0 = model_out_h_0.cuda(config['cuda_num'])
        con_lens_var = con_lens_var.cuda(config['cuda_num'])

    c_emb = embedding_layer(context, emb_h_0, context_lengths, step, name='C')
    if config['question_alone']:
        q_emb = q_emb_layer(question, emb_h_0, question_lengths, step, name='Q')
    else:
        q_emb = embedding_layer(question, emb_h_0, question_lengths, step, q_word_embedding, 'Q')
    G = att_layer(c_emb, q_emb, context_lengths, question_lengths, step)
    M = model_layer(model_out_h_0, G, context_lengths, step)
    if config['not_pretrain']:
        M_trans = M
        G_trans = G
    else:
        M_trans, G_trans = ner_hw_layer(M, G)
    prob = ner_out_layer(M_trans, G_trans, context_lengths)
    prob_size = prob.size()
    mask = Variable(get_source_mask(prob_size[0], prob_size[2], prob_size[1], context_lengths))
    mask = mask.transpose(0, 1)
    if config['USE_CUDA']:
        mask = mask.cuda(context.get_device())
    prob = prob * mask
    crf_mask = Variable(
        get_target_mask(this_batch_num, max(context_lengths), context_lengths))
    if config['USE_CUDA']:
        crf_mask = crf_mask.type(torch.cuda.ByteTensor)
        crf_mask = crf_mask.cuda(config['cuda_num'])
    else:
        crf_mask = crf_mask.type(torch.ByteTensor)
    loss = crf.neg_log_likelihood(prob.transpose(0, 1).contiguous(), target.transpose(0, 1), crf_mask, context_lengths)
    # loss = masked_cross_entropy(prob, target, con_lens_var)
    if step % 100 == 0:
        print('loss: ', loss.data[0])
    logger.scalar_summary('loss', loss.data[0], step)
    loss.backward()
    # e_before_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    # a_before_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    # m_before_step = [(tag, to_np(value)) for tag, value in model_layer.named_parameters()]
    # h_before_step = [(tag, to_np(value)) for tag, value in ner_hw_layer.named_parameters()]
    # n_before_step = [(tag, to_np(value)) for tag, value in ner_out_layer.named_parameters()]
    # c_before_step = [(tag, to_np(value)) for tag, value in crf.named_parameters()]
    # q_before_step = [(tag, to_np(value)) for tag, value in q_emb_layer.named_parameters()]


    clip_grad_norm(embedding_layer.parameters(), config['clip_norm'])
    clip_grad_norm(att_layer.parameters(), config['clip_norm'])
    clip_grad_norm(model_layer.parameters(), config['clip_norm'])
    clip_grad_norm(ner_hw_layer.parameters(), config['clip_norm'])
    clip_grad_norm(ner_out_layer.parameters(), config['clip_norm'])
    clip_grad_norm(crf.parameters(), config['clip_norm'])
    if config['question_alone']:
        clip_grad_norm(q_emb_layer.parameters(), config['clip_norm'])
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
    if not config['freeze']:
        emb_opt.step()
        att_opt.step()
        model_opt.step()
    ner_hw_opt.step()
    ner_out_opt.step()
    crf_opt.step()
    if config['question_alone']:
        q_emb_opt.step()

    grad_ratio_lst = []

    # e_after_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    # grad_ratio_lst.append((e_before_step, e_after_step))
    # a_after_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    # grad_ratio_lst.append((a_before_step, a_after_step))
    # m_after_step = [(tag, to_np(value)) for tag, value in model_layer.named_parameters()]
    # grad_ratio_lst.append((m_before_step, m_after_step))
    # h_after_step = [(tag, to_np(value)) for tag, value in ner_hw_layer.named_parameters()]
    # grad_ratio_lst.append((h_before_step, h_after_step))
    # n_after_step = [(tag, to_np(value)) for tag, value in ner_out_layer.named_parameters()]
    # grad_ratio_lst.append((n_before_step, n_after_step))
    # c_after_step = [(tag, to_np(value)) for tag, value in crf.named_parameters()]
    # grad_ratio_lst.append((c_before_step, c_after_step))
    # q_after_step = [(tag, to_np(value)) for tag, value in q_emb_layer.named_parameters()]
    #
    # utils.log_grad_ratio(logger, step, grad_ratio_lst)


def main(my_arg):
    log_dir = 'ner_logs'+str(my_arg)
    logger = Logger(log_dir)
    emb = LoadEmbedding('res/embedding.txt')
    if config['label_emb'] or config['question_alone']:
        onto_emb = LoadEmbedding('res/onto_embedding.txt')
    print('finish loading embedding')
    # batch_getter = BatchGetter('data/train', 'GPE_NAM', config['batch_size'])
    batch_getter_lst = []
    if config['bioes']:
        if config['data'] == 'conll':
            # pernam_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'PER', 1, True)
            pernam_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'PER', 1, True)
            batch_getter_lst.append(pernam_batch_getter)

            # loc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'LOC', 1, True)
            loc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'LOC', 1, True)
            batch_getter_lst.append(loc_batch_getter)

            if not config['drop_misc']:
                # misc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'MISC', 1, True)
                misc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'MISC', 1, True)
                batch_getter_lst.append(misc_batch_getter)

            # org_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'ORG', 1, True)
            org_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.train', 'ORG', 1, True)
            batch_getter_lst.append(org_batch_getter)
        elif config['data'] == 'OntoNotes':
            # onto_notes_data = TrainOntoNotesGetter('data/OntoNotes/train.json', 1, True)
            onto_notes_data = OntoNotesGetter('data/OntoNotes/leaf_train.json', ['/person', '/organization', '/location',
                                                                           '/other'], 1, True)
            batch_getter_lst.append(onto_notes_data)
    else:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'PER', 1, True)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'LOC', 1, True)
        batch_getter_lst.append(loc_batch_getter)

        if not config['drop_misc']:
            misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'MISC', 1, True)
            batch_getter_lst.append(misc_batch_getter)

        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'ORG', 1, True)
        batch_getter_lst.append(org_batch_getter)

    batch_getter = MergeBatchGetter(batch_getter_lst, config['batch_size'], True, data_name=config['data'])
    print('finish loading train data')
    # if config['data'] == 'OntoNotes':
    #     emb_onto = True
    # else:
    #     emb_onto = False
    embedding_layer = EmbeddingLayer(emb)
    if config['label_emb']:
        q_word_embedding = nn.Embedding(onto_emb.get_voc_size(), onto_emb.get_emb_size())
        q_word_embedding.weight.data.copy_(onto_emb.get_embedding_tensor())
        q_word_embedding.weight.requires_grad = False
    else:
        q_word_embedding = None
    d = config['hidden_size']
    if config['question_alone']:
        q_emb_layer = QLabel(onto_emb)
    else:
        q_emb_layer = None
    att_layer = AttentionFlowLayer(2 * d)

    model_layer = ModelingLayer(8 * d, d, 2)
    ner_hw_layer = NerHighway(2*d, 8*d, 1)
    ner_out_layer = NerOutLayer(10*d, len(config['Tags']))
    crf = CRF(config, config['Tags'], 10*d)

    if config['USE_CUDA']:
        att_layer.cuda(config['cuda_num'])
        embedding_layer.cuda(config['cuda_num'])
        if config['label_emb']:
            q_word_embedding.cuda(config['cuda_num'])
        model_layer.cuda(config['cuda_num'])
        ner_hw_layer.cuda(config['cuda_num'])
        ner_out_layer.cuda(config['cuda_num'])
        crf.cuda(config['cuda_num'])
        if config['question_alone']:
            q_emb_layer.cuda(config['cuda_num'])

    squad_model_dir = 'mr_model1'


    if not config['not_pretrain']:
        att_layer.load_state_dict(torch.load(squad_model_dir + '/early_att_layer.pkl', map_location=lambda storage, loc: storage))
        model_layer.load_state_dict(
            torch.load(squad_model_dir + '/early_model_layer.pkl', map_location=lambda storage, loc: storage))
        embedding_layer.load_state_dict(
            torch.load(squad_model_dir + '/early_embedding_layer.pkl', map_location=lambda storage, loc: storage))



    if config['freeze']:
        for param in att_layer.parameters():
            param.requires_grad = False
        for param in model_layer.parameters():
            param.requires_grad = False
        for param in embedding_layer.parameters():
            param.requires_grad = False
        embedding_layer.eval()
        model_layer.eval()
        att_layer.eval()
        emb_opt = None
        att_opt = None
        model_opt = None
    else:
        if config['not_pretrain']:
            emb_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, embedding_layer.parameters()))
            att_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, att_layer.parameters()))
            model_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, model_layer.parameters()))
        else:
            emb_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, embedding_layer.parameters()), lr=1e-4)
            att_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, att_layer.parameters()), lr=1e-4)
            model_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, model_layer.parameters()), lr=1e-4)

    # model_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, model_layer.parameters()))
    ner_hw_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, ner_hw_layer.parameters()))
    ner_out_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, ner_out_layer.parameters()))
    crf_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, crf.parameters()))
    if config['question_alone']:
        q_emb_opt = torch.optim.Adam(filter(lambda param: param.requires_grad, q_emb_layer.parameters()))
    else:
        q_emb_opt = None

    log_file = open('{}/log_file'.format(log_dir), 'w')
    f_max = 0
    low_epoch = 0
    ex_iterations = 0
    model_dir = 'ner_model'+str(my_arg)
    time0 = time.time()
    for epoch in range(config['max_epoch']):
        embedding_layer.train()
        att_layer.train()
        model_layer.train()
        ner_hw_layer.train()
        ner_out_layer.train()
        crf.train()
        if config['question_alone']:
            q_emb_layer.train()
        # f, p, r = evaluate_all(my_arg, False)
        for iteration, this_batch in enumerate(batch_getter):
            if (ex_iterations + iteration) % 100 == 0:
                print('epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration))

            train_iteration(logger, ex_iterations + iteration, embedding_layer, q_word_embedding, q_emb_layer, att_layer, model_layer, ner_hw_layer, ner_out_layer, crf,
                            emb_opt, q_emb_opt, att_opt, model_opt, ner_hw_opt, ner_out_opt, crf_opt, this_batch)
            if (ex_iterations + iteration) % 100 == 0:
                time1 = time.time()
                print('this iteration time: ', time1 - time0, '\n')
                time0 = time1
            if (ex_iterations + iteration) % config['save_freq'] == 0:
                torch.save(embedding_layer.state_dict(), model_dir+'/embedding_layer.pkl')
                torch.save(att_layer.state_dict(), model_dir+'/att_layer.pkl')
                torch.save(model_layer.state_dict(), model_dir+'/model_layer.pkl')
                torch.save(ner_hw_layer.state_dict(), model_dir+'/ner_hw_layer.pkl')
                torch.save(ner_out_layer.state_dict(), model_dir + '/ner_out_layer.pkl')
                torch.save(crf.state_dict(), model_dir+'/crf.pkl')
                if config['question_alone']:
                    torch.save(q_emb_layer.state_dict(), model_dir+'/q_emb_layer.pkl')


        ex_iterations += iteration + 1
        batch_getter.reset()
        config['use_dropout'] = False
        f, p, r = evaluate_all(my_arg, False)
        config['use_dropout'] = True
        log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(epoch, f, p, r))
        log_file.flush()
        if f >= f_max:
            f_max = f
            low_epoch = 0
            os.system('cp {}/embedding_layer.pkl {}/early_embedding_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/att_layer.pkl {}/early_att_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/model_layer.pkl {}/early_model_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/ner_hw_layer.pkl {}/early_ner_hw_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/ner_out_layer.pkl {}/early_ner_out_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/crf.pkl {}/early_crf.pkl'.format(model_dir, model_dir))
            if config['question_alone']:
                os.system('cp {}/q_emb_layer.pkl {}/early_q_emb_layer.pkl'.format(model_dir, model_dir))

        else:
            low_epoch += 1
            log_file.write('low' + str(low_epoch) + '\n')
            log_file.flush()
        if low_epoch >= config['early_stop']:
            break
    log_file.close()

if __name__ == '__main__':
    # print get_question(3)
    # my_arg = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--my_arg', help='mode', type=int)
    parser.add_argument('--batch', help='batch', type=int)
    parser.add_argument('--use_gaz', action='store_true', help='verbose mode')
    parser.add_argument('--cuda_num', help='cuda_num', type=int)
    parser.add_argument('--transition', action='store_true', help='whether set start and end of transition matrix -1e30')
    parser.add_argument('--alone', action='store_true', help='whether set question weight alone')
    parser.add_argument('--bioes', action='store_true', help='bioes')
    parser.add_argument('--misc', action='store_true', help='misc')
    parser.add_argument('--drop_misc', action='store_true', help='drop_misc')
    parser.add_argument('--sigmoid', action='store_true', help='sigmoid')
    parser.add_argument('--entity_emb', action='store_true', help='entity_emb')
    parser.add_argument('--gate', action='store_true', help='gate')
    parser.add_argument('--freeze', action='store_true', help='freeze params of embedding, att_lay, model_layer')
    parser.add_argument('--large_crf', action='store_true', help='verbose mode')
    parser.add_argument('--data', help='conll or OntoNotes', default='conll')
    parser.add_argument('--not_pretrain', action='store_true', help='do not use mr pretrained model')
    parser.add_argument('--label_emb', action='store_true', help='use prototype label embedding')
    args = parser.parse_args()
    my_arg = args.my_arg
    batch_size = args.batch
    # my_arg = 2
    use_gaz = args.use_gaz
    cuda_num = args.cuda_num
    transition = args.transition
    alone = args.alone
    bioes = args.bioes
    config['use_gaz'] = use_gaz
    config['cuda_num'] = cuda_num
    config['batch_size'] = batch_size
    config['transition'] = transition
    config['question_alone'] = alone
    config['bioes'] = bioes
    config['misc'] = args.misc
    config['drop_misc'] = args.drop_misc
    config['sigmoid'] = args.sigmoid
    config['entity_emb'] = args.entity_emb
    config['gate'] = args.gate
    config['freeze'] = args.freeze
    config['large_crf'] = args.large_crf
    config['data'] = args.data
    config['not_pretrain'] = args.not_pretrain
    config['label_emb'] = args.label_emb

    if config['use_gaz']:
        gazdir = config['GazetteerDir']
        gaz_names = config['Gazetteers']
        gazetteers = []
        for (id, gaz) in enumerate(gaz_names):
            gazfile = os.path.join(gazdir, gaz)
            gazetteers.append(utils.load_gaz_list(gazfile))
        config['load_gaz'] = gazetteers

    if config['bioes']:
        config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4, 'E': 5, 'S': 6}
    else:
        config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4}
    torch.cuda.set_device(config['cuda_num'])
    main(my_arg)
