import torch
from configurations import config, to_np
from logger import Logger
from batch_getter import BatchGetter, get_question, MergeBatchGetter
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingOutLayer

from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
from eval_ner import evaluate_all
from numpy import linalg as LA
import sys
import os
from conll_data_trans import ConllBatchGetter
import argparse
# import pydevd
#
# pydevd.settrace('10.214.129.230', port=31235, stdoutToServer=True, stderrToServer=True)
torch.manual_seed(0)





def train_iteration(logger, step, embedding_layer, att_layer, model_out_layer, emb_opt, att_opt, model_out_opt, this_batch):
    emb_opt.zero_grad()
    att_opt.zero_grad()
    model_out_opt.zero_grad()

    d = embedding_layer.out_dim
    this_batch_num = len(this_batch[2])

    # question = Variable(get_question('%GPE%', this_batch_num))  # (batch, J=1, 51)
    question = Variable(this_batch[4])
    # question_lengths = [1 for _ in range(this_batch_num)]
    question_lengths = this_batch[5]
    context = Variable(this_batch[0])  # (batch, T, 51)
    context_lengths = this_batch[3]  # list
    target = Variable(this_batch[1])  # (batch, T)
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d))
    model_out_h_0 = Variable(torch.zeros(2*model_out_layer.num_layers, this_batch_num, d))
    con_lens_var = Variable(torch.LongTensor(context_lengths))

    if config['USE_CUDA']:
        question = question.cuda(config['cuda_num'])
        context = context.cuda(config['cuda_num'])
        target = target.cuda(config['cuda_num'])
        emb_h_0 = emb_h_0.cuda(config['cuda_num'])
        model_out_h_0 = model_out_h_0.cuda(config['cuda_num'])
        con_lens_var = con_lens_var.cuda(config['cuda_num'])

    c_emb = embedding_layer(context, emb_h_0, context_lengths, step, 'C')
    q_emb = embedding_layer(question, emb_h_0, question_lengths, step, 'Q')
    G = att_layer(c_emb, q_emb, context_lengths, question_lengths, step)
    prob = model_out_layer(model_out_h_0, G, context_lengths, step)
    loss = masked_cross_entropy(prob, target, con_lens_var)
    print 'loss: ', loss.data[0]
    logger.scalar_summary('loss', loss.data[0], step)
    loss.backward()
    e_before_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    a_before_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    m_before_step = [(tag, to_np(value)) for tag, value in model_out_layer.named_parameters()]


    clip_grad_norm(embedding_layer.parameters(), config['clip_norm'])
    clip_grad_norm(att_layer.parameters(), config['clip_norm'])
    clip_grad_norm(model_out_layer.parameters(), config['clip_norm'])
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
    emb_opt.step()
    att_opt.step()
    model_out_opt.step()

    e_after_step = [(tag, to_np(value)) for tag, value in embedding_layer.named_parameters()]
    a_after_step = [(tag, to_np(value)) for tag, value in att_layer.named_parameters()]
    m_after_step = [(tag, to_np(value)) for tag, value in model_out_layer.named_parameters()]

    for before, after in zip(e_before_step, e_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)

    for before, after in zip(a_before_step, a_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)
    for before, after in zip(m_before_step, m_after_step):
        if before[0] == after[0]:
            tag = before[0]
            value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
            tag = tag.replace('.', '/')
            if value is not None:
                logger.scalar_summary(tag + '/grad_ratio', value, step)


def main(my_arg):
    logger = Logger('./logs'+str(my_arg))
    emb = LoadEmbedding('res/embedding.txt')
    print 'finish loading embedding'
    # batch_getter = BatchGetter('data/train', 'GPE_NAM', config['batch_size'])
    batch_getter_lst = []
    if my_arg == 0:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'PER', 1, True)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'LOC', 1, True)
        batch_getter_lst.append(loc_batch_getter)

        misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'MISC', 1, True)
        batch_getter_lst.append(misc_batch_getter)

        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'ORG', 1, True)
        batch_getter_lst.append(org_batch_getter)

    if my_arg == 1:
        pernam_batch_getter = BatchGetter('data/train', 'PER_NAM', 1)
        batch_getter_lst.append(pernam_batch_getter)

        fac_batch_getter = BatchGetter('data/train', 'FAC_NAM', 1)
        batch_getter_lst.append(fac_batch_getter)

        loc_batch_getter = BatchGetter('data/train', 'LOC_NAM', 1)
        batch_getter_lst.append(loc_batch_getter)

        gpe_batch_getter = BatchGetter('data/train', 'GPE_NAM', 1)
        batch_getter_lst.append(gpe_batch_getter)

        # if my_arg == 1:
        org_batch_getter = BatchGetter('data/train', 'ORG_NAM', 1)
        batch_getter_lst.append(org_batch_getter)
    if my_arg == 2:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'PER', 1, True)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'LOC', 1, True)
        batch_getter_lst.append(loc_batch_getter)

        misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'MISC', 1, True)
        batch_getter_lst.append(misc_batch_getter)

        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.train', 'ORG', 1, True)
        batch_getter_lst.append(org_batch_getter)

    batch_getter = MergeBatchGetter(batch_getter_lst, config['batch_size'], True)
    print 'finish loading train data'
    embedding_layer = EmbeddingLayer(emb)
    d = embedding_layer.get_out_dim()
    att_layer = AttentionFlowLayer(2 * d)
    # if my_arg == 2:
    model_out_layer = ModelingOutLayer(8 * d, d, 2, 3)
    # else:
    #     model_out_layer = ModelingOutLayer(8*d, d, 2, 2)
    # models = [embedding_layer, att_layer, model_out_layer]
    # opts = [emb_opt, att_opt, model_out_opt]

    if config['USE_CUDA']:
        att_layer.cuda(config['cuda_num'])
        embedding_layer.cuda(config['cuda_num'])
        model_out_layer.cuda(config['cuda_num'])

    emb_opt = torch.optim.Adadelta(embedding_layer.parameters())
    att_opt = torch.optim.Adadelta(att_layer.parameters())
    model_out_opt = torch.optim.Adadelta(model_out_layer.parameters())

    log_file = open('log_file'+str(my_arg), 'w')
    f_max = 0
    low_epoch = 0
    ex_iterations = 0
    model_dir = 'model'+str(my_arg)
    for epoch in range(config['max_epoch']):
        for iteration, this_batch in enumerate(batch_getter):
            time0 = time.time()
            print 'epoch: {}, iteraton: {}'.format(epoch, ex_iterations + iteration)
            train_iteration(logger, ex_iterations + iteration, embedding_layer, att_layer, model_out_layer, emb_opt, att_opt, model_out_opt, this_batch)
            time1 = time.time()
            print 'this iteration time: ', time1 - time0, '\n'
            if (ex_iterations + iteration) % config['save_freq'] == 0:
                torch.save(embedding_layer.state_dict(), model_dir+'/embedding_layer.pkl')
                torch.save(att_layer.state_dict(), model_dir+'/att_layer.pkl')
                torch.save(model_out_layer.state_dict(), model_dir+'/model_out_layer.pkl')

        ex_iterations += iteration + 1
        batch_getter.reset()
        f, p, r = evaluate_all(my_arg, False)
        log_file.write('epoch: {} f: {} p: {} r: {}\n'.format(epoch, f, p, r))
        log_file.flush()
        if f >= f_max:
            f_max = f
            low_epoch = 0
            # torch.save(embedding_layer.state_dict(), model_dir+'/early_embedding_layer.pkl')
            # torch.save(att_layer.state_dict(), model_dir+'/early_att_layer.pkl')
            # torch.save(model_out_layer.state_dict(), model_dir+'/early_model_out_layer.pkl')
            os.system('cp {}/embedding_layer.pkl {}/early_embedding_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/att_layer.pkl {}/early_att_layer.pkl'.format(model_dir, model_dir))
            os.system('cp {}/model_out_layer.pkl {}/early_model_out_layer.pkl'.format(model_dir, model_dir))

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
    parser.add_argument('--use_gaz', action='store_true', help='verbose mode')
    args = parser.parse_args()
    my_arg = args.my_arg
    # my_arg = 2
    use_gaz = args.use_gaz
    print type(my_arg), my_arg
    print type(use_gaz), use_gaz
    config['use_gaz'] = use_gaz
    main(my_arg)
