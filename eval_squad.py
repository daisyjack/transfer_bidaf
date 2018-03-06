from __future__ import print_function
import torch
from configurations import config, to_np, Vocab
from batch_getter import BatchGetter, get_question, MergeBatchGetter
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingLayer, StartProbLayer, EndProbLayer, VERY_NEGATIVE_NUMBER

from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
import codecs
from conll_data_trans import ConllBatchGetter, ConllBoundaryPerformance
from crf import CRF
from batch_getter import get_target_mask, get_source_mask
# from itertools import ifilter
from tensorboardX import SummaryWriter
from squad_dataloader import SquadLoader, remove_puctuation
import json
import sys
from official_evaluate import evaluate
torch.manual_seed(0)

writer = SummaryWriter('runs/exp9')


def diag_logits_exp_mask(val):
    mask = Variable(torch.zeros(val.size(1), val.size(1)))
    if config['USE_CUDA']:
        mask = mask.cuda(val.get_device())
    for i in range(1, val.size(1)):
        mask[i, 0:i] = 1
    mask = mask.unsqueeze(0).expand(val.size(0), val.size(1), val.size(1))
    return mask * VERY_NEGATIVE_NUMBER






def evaluate_batch(step, embedding_layer, att_layer, model_layer, start_layer, end_layer, this_batch,
                 summary_emb=False, all_emb=None, all_metadata=None):
    d = config['hidden_size']
    this_batch_num = len(this_batch['con_lens'])
    question = Variable(this_batch['questions'], volatile=True)
    question_lengths = this_batch['q_lens']
    context = Variable(this_batch['contexts'], volatile=True)  # (batch, T, 51)
    context_lengths = this_batch['con_lens']  # list

    start_target = Variable(this_batch['start'], volatile=True)
    end_target = Variable(this_batch['end'], volatile=True)
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d), volatile=True)
    model_h_0 = Variable(torch.zeros(2 * model_layer.num_layers, this_batch_num, d), volatile=True)
    end_h_0 = Variable(torch.zeros(2, this_batch_num, d), volatile=True)

    if config['USE_CUDA']:
        question = question.cuda(config['cuda_num'])
        context = context.cuda(config['cuda_num'])
        emb_h_0 = emb_h_0.cuda(config['cuda_num'])
        model_h_0 = model_h_0.cuda(config['cuda_num'])
        end_h_0 = end_h_0.cuda(config['cuda_num'])
        start_target = start_target.cuda(config['cuda_num'])
        end_target = end_target.cuda(config['cuda_num'])

    c_emb = embedding_layer(context, emb_h_0, context_lengths, step,
                            'C')  # (seq_len, batch, hidden_size(d=100) * num_directions(2))
    q_emb = embedding_layer(question, emb_h_0, question_lengths, step,
                            'Q')  # (seq_len, batch, hidden_size(d=100) * num_directions(2))
    G = att_layer(c_emb, q_emb, context_lengths, question_lengths, step)  # (batch, T, 8d)
    M = model_layer(model_h_0, G, context_lengths, step)  # M: (batch, T, 2d)
    start_logits = start_layer(M, G, context_lengths)  # (batch, T)
    end_logits = end_layer(M, G, end_h_0, context_lengths)  # (batch, T)
    diag_start_logits = start_logits.unsqueeze(2).expand(start_logits.size(0), start_logits.size(1), start_logits.size(1))
    diag_end_logits = end_logits.unsqueeze(1).expand(end_logits.size(0), end_logits.size(1), end_logits.size(1))
    diag_logits_mask = diag_logits_exp_mask(start_logits)
    mat = diag_start_logits + diag_end_logits + diag_logits_mask
    val_0, index_0 = torch.max(mat, 2)  # (batch, T)
    val_1, start_index = torch.max(val_0, 1)  # (batch,)
    end_index = torch.gather(index_0, 1, start_index.unsqueeze(1))  # (batch, 1)
    end_index = end_index.squeeze(1)
    return start_index, end_index


def evaluate_all(my_arg, pr=True):
    emb = LoadEmbedding('res/emb.txt')
    print('finish loading embedding')
    batch_size = 100
    batch_getter = SquadLoader('data/SQuAD/dev-v1.1.json', batch_size, False)
    print('finish loading dev data')
    embedding_layer = EmbeddingLayer(emb, dropout_p=0)
    d = config['hidden_size']
    att_layer = AttentionFlowLayer(2 * d)
    model_layer = ModelingLayer(8 * d, d, 2, dropout=0)
    start_layer = StartProbLayer(10 * d, dropout=0)
    end_layer = EndProbLayer(2 * d, d, dropout=0)

    if config['USE_CUDA']:
        att_layer.cuda(config['cuda_num'])
        embedding_layer.cuda(config['cuda_num'])
        model_layer.cuda(config['cuda_num'])
        start_layer.cuda(config['cuda_num'])
        end_layer.cuda(config['cuda_num'])
    model_dir = 'mr_model' + str(my_arg)

    embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    att_layer.load_state_dict(torch.load(model_dir+'/att_layer.pkl'))
    model_layer.load_state_dict(torch.load(model_dir+'/model_layer.pkl'))
    start_layer.load_state_dict(torch.load(model_dir+'/start_layer.pkl'))
    end_layer.load_state_dict(torch.load(model_dir+'/end_layer.pkl'))

    embedding_layer.eval()
    att_layer.eval()
    model_layer.eval()
    start_layer.eval()
    end_layer.eval()

    result_json = {}

    ex_iterations = 0
    for iteration, this_batch in enumerate(batch_getter):
        start_index, end_index = evaluate_batch(ex_iterations + iteration, embedding_layer, att_layer,
                                                model_layer, start_layer, end_layer, this_batch)
        start_cpu = start_index.cpu().data.numpy()
        end_cpu = end_index.cpu().data.numpy()
        this_batch_size = len(this_batch['ids'])
        for i in range(this_batch_size):
            # start_num = this_batch['ans_start'][i]
            start_num = start_cpu[i]
            # end_num = this_batch['ans_end'][i]
            end_num = end_cpu[i]
            q_id = this_batch['ids'][i]
            art_id = this_batch['art_ids'][i]
            para_id = this_batch['para_ids'][i]
            context = batch_getter.dataset[art_id]['paragraphs'][para_id]['context']
            ans_word_lst = context.split()[start_num:end_num+1]
            # ans_word_lst[-1] = remove_puctuation(ans_word_lst[-1])
            ans = ' '.join(ans_word_lst)
            result_json[q_id] = ans
        if (iteration+1)*batch_size % 100 == 0:
            print('{} questions processed'.format((iteration+1)*batch_size))

    with open('data/squad_pred'+str(my_arg), mode='w') as out_f:
        json.dump(result_json, out_f)

    expected_version = '1.1'
    with open('data/SQuAD/dev-v1.1.json') as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    predictions = result_json
    r = evaluate(dataset, predictions)
    print(r)
    return r['exact_match'], r['f1']






if __name__ == '__main__':

    def evaluate_free(my_arg, pr=True):
        emb = LoadEmbedding('res/emb.txt')
        print('finish loading embedding')
        batch_size = 100
        batch_getter = SquadLoader('data/SQuAD/dev-v1.1.json', batch_size, False)
        # batch_getter = SquadLoader('data/SQuAD/train-v1.1.json', batch_size, False)
        print('finish loading dev data')
        embedding_layer = EmbeddingLayer(emb, dropout_p=0)
        d = config['hidden_size']
        att_layer = AttentionFlowLayer(2 * d)
        model_layer = ModelingLayer(8 * d, d, 2, dropout=0)
        start_layer = StartProbLayer(10 * d, dropout=0)
        end_layer = EndProbLayer(2 * d, d, dropout=0)

        if config['USE_CUDA']:
            att_layer.cuda(config['cuda_num'])
            embedding_layer.cuda(config['cuda_num'])
            model_layer.cuda(config['cuda_num'])
            start_layer.cuda(config['cuda_num'])
            end_layer.cuda(config['cuda_num'])
        model_dir = 'mr_model' + str(my_arg)

        embedding_layer.load_state_dict(torch.load(model_dir + '/embedding_layer.pkl', map_location=lambda storage, loc: storage))
        att_layer.load_state_dict(torch.load(model_dir + '/att_layer.pkl', map_location=lambda storage, loc: storage))
        model_layer.load_state_dict(torch.load(model_dir + '/model_layer.pkl', map_location=lambda storage, loc: storage))
        start_layer.load_state_dict(torch.load(model_dir + '/start_layer.pkl', map_location=lambda storage, loc: storage))
        end_layer.load_state_dict(torch.load(model_dir + '/end_layer.pkl', map_location=lambda storage, loc: storage))
        # a = torch.load(model_dir+'/opt.pkl', map_location=lambda storage, loc: storage)
        # print(a['iteration'])
        embedding_layer.eval()
        att_layer.eval()
        model_layer.eval()
        start_layer.eval()
        end_layer.eval()

        result_json = {}

        ex_iterations = 0
        for iteration, this_batch in enumerate(batch_getter):
            start_index, end_index = evaluate_batch(ex_iterations + iteration, embedding_layer, att_layer,
                                                    model_layer, start_layer, end_layer, this_batch)
            start_cpu = start_index.cpu().data.numpy()
            end_cpu = end_index.cpu().data.numpy()
            this_batch_size = len(this_batch['ids'])
            for i in range(this_batch_size):
                # start_num = this_batch['ans_start'][i]
                start_num = start_cpu[i]
                # end_num = this_batch['ans_end'][i]
                end_num = end_cpu[i]
                q_id = this_batch['ids'][i]
                art_id = this_batch['art_ids'][i]
                para_id = this_batch['para_ids'][i]
                context = batch_getter.dataset[art_id]['paragraphs'][para_id]['context']
                ans_word_lst = context.split()[start_num:end_num + 1]
                # ans_word_lst[-1] = remove_puctuation(ans_word_lst[-1])
                ans = ' '.join(ans_word_lst)
                result_json[q_id] = ans
            if (iteration + 1) * batch_size % 100 == 0:
                print('{} questions processed'.format((iteration + 1) * batch_size))

        with open('data/squad_pred' + str(my_arg), mode='w') as out_f:
            json.dump(result_json, out_f)

        expected_version = '1.1'
        with open('data/SQuAD/dev-v1.1.json') as dataset_file:
            dataset_json = json.load(dataset_file)
            if dataset_json['version'] != expected_version:
                print('Evaluation expects v-' + expected_version +
                      ', but got dataset with v-' + dataset_json['version'],
                      file=sys.stderr)
            dataset = dataset_json['data']
        predictions = result_json
        r = evaluate(dataset, predictions)
        print(r)
        return r['exact_match'], r['f1']

    config['cuda_num'] = 1
    config['batch_size'] = 100
    # config['dropout'] = args.drop_out
    config['gate'] = False
    config['sigmoid'] = False
    config['use_gaz'] = False
    config['use_dropout'] = False
    torch.cuda.set_device(config['cuda_num'])
    evaluate_free(1, False)
