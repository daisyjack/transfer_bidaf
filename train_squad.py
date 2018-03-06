from __future__ import print_function
import torch
from configurations import config, to_np
from logger import Logger
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingLayer, StartProbLayer, EndProbLayer
from crf import CRF
from torch.autograd import Variable
import numpy
from torch.nn.utils import clip_grad_norm
from eval_squad import evaluate_all
from numpy import linalg as LA
import argparse
import os
from batch_getter import get_source_mask, get_target_mask
from squad_dataloader import SquadLoader
import random
# from itertools import ifilter
# import pydevd
#
# pydevd.settrace('10.214.129.230', port=31235, stdoutToServer=True, stderrToServer=True)
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)





def train_iteration(logger, step, embedding_layer, att_layer, model_layer, start_layer, end_layer,
                    emb_opt, att_opt, model_opt, start_opt, end_opt, this_batch):
    emb_opt.zero_grad()
    att_opt.zero_grad()
    model_opt.zero_grad()
    start_opt.zero_grad()
    end_opt.zero_grad()

    d = config['hidden_size']
    this_batch_num = len(this_batch['con_lens'])
    question = Variable(this_batch['questions'])
    question_lengths = this_batch['q_lens']
    context = Variable(this_batch['contexts'])  # (batch, T, 51)
    context_lengths = this_batch['con_lens']  # list


    start_target = Variable(this_batch['start'])
    end_target = Variable(this_batch['end'])
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d))
    model_h_0 = Variable(torch.zeros(2*model_layer.num_layers, this_batch_num, d))
    end_h_0 = Variable(torch.zeros(2, this_batch_num, d))

    if config['USE_CUDA']:
        question = question.cuda(config['cuda_num'])
        context = context.cuda(config['cuda_num'])
        emb_h_0 = emb_h_0.cuda(config['cuda_num'])
        model_h_0 = model_h_0.cuda(config['cuda_num'])
        end_h_0 = end_h_0.cuda(config['cuda_num'])
        start_target = start_target.cuda(config['cuda_num'])
        end_target = end_target.cuda(config['cuda_num'])

    c_emb = embedding_layer(context, emb_h_0, context_lengths, step, 'C')  # (seq_len, batch, hidden_size(d=100) * num_directions(2))
    q_emb = embedding_layer(question, emb_h_0, question_lengths, step, 'Q')  # (seq_len, batch, hidden_size(d=100) * num_directions(2))
    G = att_layer(c_emb, q_emb, context_lengths, question_lengths, step)  # (batch, T, 8d)
    M = model_layer(model_h_0, G, context_lengths, step)  # M: (batch, T, 2d)
    start_logits = start_layer(M, G, context_lengths)  # (batch, T)
    end_logits = end_layer(M, G, end_h_0, context_lengths)  # (batch, T)
    loss = -torch.sum(start_logits*start_target+end_logits*end_target) / this_batch_num
    print('loss: ', loss.data[0])
    logger.scalar_summary('loss', loss.data[0], step)
    loss.backward()
    # e_before_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    # a_before_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    # m_before_step = [(tag, to_np(value)) for tag, value in model_layer.named_parameters()]
    # start_before_step = [(tag, to_np(value)) for tag, value in start_layer.named_parameters()]
    # end_before_step = [(tag, to_np(value)) for tag, value in end_layer.named_parameters()]


    clip_grad_norm(embedding_layer.parameters(), config['clip_norm'])
    clip_grad_norm(att_layer.parameters(), config['clip_norm'])
    clip_grad_norm(model_layer.parameters(), config['clip_norm'])
    clip_grad_norm(start_layer.parameters(), config['clip_norm'])
    clip_grad_norm(end_layer.parameters(), config['clip_norm'])
    for tag, value in embedding_layer.named_parameters():
        tag = tag.replace('.', '/')
        if value is not None and value.grad is not None:
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    for tag, value in att_layer.named_parameters():
        tag = tag.replace('.', '/')
        if value is not None and value.grad is not None:
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    for tag, value in model_layer.named_parameters():
        tag = tag.replace('.', '/')
        if value is not None and value.grad is not None:
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    for tag, value in start_layer.named_parameters():
        tag = tag.replace('.', '/')
        if value is not None and value.grad is not None:
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    for tag, value in end_layer.named_parameters():
        tag = tag.replace('.', '/')
        if value is not None and value.grad is not None:
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
    emb_opt.step()
    att_opt.step()
    model_opt.step()
    start_opt.step()
    end_opt.step()

    # e_after_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    # a_after_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    # m_after_step = [(tag, to_np(value)) for tag, value in model_layer.named_parameters()]
    # start_after_step = [(tag, to_np(value)) for tag, value in start_layer.named_parameters()]
    # end_after_step = [(tag, to_np(value)) for tag, value in end_layer.named_parameters()]


    # for before, after in zip(e_before_step, e_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)
    #
    # for before, after in zip(a_before_step, a_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)
    # for before, after in zip(m_before_step, m_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)
    # for before, after in zip(start_before_step, start_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)
    # for before, after in zip(end_before_step, end_after_step):
    #     if before[0] == after[0]:
    #         tag = before[0]
    #         value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
    #         tag = tag.replace('.', '/')
    #         if value is not None:
    #             logger.scalar_summary(tag + '/grad_ratio', value, step)


def main(my_arg):
    log_dir = 'mr_logs'+str(my_arg)
    logger = Logger(log_dir)
    emb = LoadEmbedding('res/embedding.txt')
    print('finish loading embedding')
    batch_getter = SquadLoader('data/SQuAD/train-v1.1.json', config['batch_size'], True)
    # batch_getter = SquadLoader('data/SQuAD/dev-v1.1.json', config['batch_size'], True)
    print('finish loading train data')
    embedding_layer = EmbeddingLayer(emb)
    d = config['hidden_size']
    att_layer = AttentionFlowLayer(2 * d)
    model_layer = ModelingLayer(8 * d, d, 2)
    start_layer = StartProbLayer(10*d)
    end_layer = EndProbLayer(2*d, d)

    if config['USE_CUDA']:
        att_layer.cuda(config['cuda_num'])
        embedding_layer.cuda(config['cuda_num'])
        model_layer.cuda(config['cuda_num'])
        start_layer.cuda(config['cuda_num'])
        end_layer.cuda(config['cuda_num'])

    emb_opt = torch.optim.Adam(embedding_layer.parameters())
    att_opt = torch.optim.Adam(att_layer.parameters())
    model_opt = torch.optim.Adam(model_layer.parameters())
    start_opt = torch.optim.Adam(start_layer.parameters())
    end_opt = torch.optim.Adam(end_layer.parameters())

    model_dir = 'mr_model' + str(my_arg)
    check_epoch = 0
    check_ex_iteration = 0

    if config['resume']:
        check = torch.load(model_dir + '/opt.pkl')
        emb_opt.load_state_dict(check['emb_opt'])
        att_opt.load_state_dict(check['att_opt'])
        model_opt.load_state_dict(check['model_opt'])
        start_opt.load_state_dict(check['start_opt'])
        end_opt.load_state_dict(check['end_opt'])
        check_epoch = check['epoch']
        check_ex_iteration = check['iteration']

        embedding_layer.load_state_dict(torch.load(model_dir + '/embedding_layer.pkl'))
        att_layer.load_state_dict(torch.load(model_dir + '/att_layer.pkl'))
        model_layer.load_state_dict(torch.load(model_dir + '/model_layer.pkl'))
        start_layer.load_state_dict(torch.load(model_dir + '/start_layer.pkl'))
        end_layer.load_state_dict(torch.load(model_dir + '/end_layer.pkl'))




    log_file = open('{}/log_file'.format(log_dir), 'w')
    f_max = 0
    low_epoch = 0
    ex_iterations = check_ex_iteration+1

    for epoch in range(check_epoch, config['epochs']):
        embedding_layer.train()
        att_layer.train()
        model_layer.train()
        start_layer.train()
        end_layer.train()
        # exact_match, f = evaluate_all(my_arg, False)
        for iteration, this_batch in enumerate(batch_getter):
            time0 = time.time()
            print('epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration))
            train_iteration(logger, ex_iterations + iteration, embedding_layer, att_layer, model_layer, start_layer, end_layer,
                            emb_opt, att_opt, model_opt, start_opt, end_opt, this_batch)
            time1 = time.time()
            print('this iteration time: ', time1 - time0, '\n')
            if (ex_iterations + iteration) % config['save_freq'] == 0:
                torch.save(embedding_layer.state_dict(), model_dir+'/embedding_layer.pkl')
                torch.save(att_layer.state_dict(), model_dir+'/att_layer.pkl')
                torch.save(model_layer.state_dict(), model_dir+'/model_layer.pkl')
                torch.save(start_layer.state_dict(), model_dir+'/start_layer.pkl')
                torch.save(end_layer.state_dict(), model_dir + '/end_layer.pkl')
                check_point = {'epoch': epoch, 'iteration': ex_iterations+iteration, 'emb_opt': emb_opt.state_dict(), 'att_opt': att_opt.state_dict(),
                               'model_opt': model_opt.state_dict(), 'start_opt': start_opt.state_dict(), 'end_opt': end_opt.state_dict()}
                torch.save(check_point, model_dir+'/opt.pkl')
        if epoch == 11:
            torch.save(embedding_layer.state_dict(), model_dir + '/12_embedding_layer.pkl')
            torch.save(att_layer.state_dict(), model_dir + '/12_att_layer.pkl')
            torch.save(model_layer.state_dict(), model_dir + '/12_model_layer.pkl')
            torch.save(start_layer.state_dict(), model_dir + '/12_start_layer.pkl')
            torch.save(end_layer.state_dict(), model_dir + '/12_end_layer.pkl')
            check_point = {'epoch': epoch, 'iteration': ex_iterations + iteration, 'emb_opt': emb_opt.state_dict(),
                           'att_opt': att_opt.state_dict(),
                           'model_opt': model_opt.state_dict(), 'start_opt': start_opt.state_dict(),
                           'end_opt': end_opt.state_dict()}
            torch.save(check_point, model_dir + '/opt.pkl')


        ex_iterations += iteration + 1
        batch_getter.reset()
        config['use_dropout'] = False
        exact_match, f = evaluate_all(my_arg, False)
        config['use_dropout'] = True
        log_file.write('epoch: {} exact_match: {} f: {}\n'.format(epoch, exact_match, f))
        log_file.flush()
        if f >= f_max:
            f_max = f
            low_epoch = 0
            os.system('cp {}/embedding_layer.pkl {}/early_embedding_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/att_layer.pkl {}/early_att_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/model_layer.pkl {}/early_model_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/start_layer.pkl {}/early_start_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/end_layer.pkl {}/early_end_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/opt.pkl {}/early_opt.pkl'.format(model_dir, model_dir))

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
    parser.add_argument('--cuda_num', help='cuda_num', type=int)
    parser.add_argument('--epochs', help='epochs', type=int)
    parser.add_argument('--resume', action='store_true', help='resume')
    # parser.add_argument('--char_emb', help='char_emb', type=int)
    # parser.add_argument('--drop_out', help='drop_out', type=float)
    args = parser.parse_args()
    my_arg = args.my_arg
    batch_size = args.batch
    cuda_num = args.cuda_num
    config['cuda_num'] = cuda_num
    config['batch_size'] = batch_size
    # config['dropout'] = args.drop_out
    config['gate'] = False
    config['sigmoid'] = False
    config['use_gaz'] = False
    config['epochs'] = args.epochs
    config['resume'] = args.resume
    config['freeze'] = False
    torch.cuda.set_device(config['cuda_num'])
    main(my_arg)
