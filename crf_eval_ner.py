from __future__ import print_function
import torch
from configurations import config, to_np, Vocab
from batch_getter import BatchGetter, get_question, MergeBatchGetter
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingLayer, QEmbeddingLayer, NerOutLayer, NerHighway, QLabel

from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
import codecs
from conll_data_trans import ConllBatchGetter, ConllBoundaryPerformance, OntoNotesGetter
from crf import CRF
from batch_getter import get_target_mask, get_source_mask
# from itertools import ifilter
from tensorboardX import SummaryWriter
import utils
from torch import nn
torch.manual_seed(0)

writer = SummaryWriter('runs/exp9')





def evaluate_one(step, embedding_layer, q_word_embedding, q_emb_layer, att_layer, model_layer, ner_hw_layer, ner_out_layer, crf, this_batch,
                 summary_emb=False, all_emb=None, all_metadata=None):

    d = config['hidden_size']
    this_batch_num = len(this_batch[2])
    question = Variable(this_batch[4])
    question_lengths = this_batch[5]
    context = Variable(this_batch[0], volatile=True)  # (batch, T, 51)
    context_lengths = this_batch[3]  # list
    target = Variable(this_batch[1], volatile=True)  # (batch, T)
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d), volatile=True)
    model_out_h_0 = Variable(torch.zeros(2*model_layer.num_layers, this_batch_num, d), volatile=True)
    con_lens_var = Variable(torch.LongTensor(context_lengths), volatile=True)

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

    if summary_emb:
        for i in range(this_batch_num):
            sentence = ''
            metadata = []
            for tokenId, token in enumerate(context[i]):
                if tokenId >= context_lengths[i]:
                    break
                word = config['WordId'].getWord(token[0].data.cpu().numpy()[0])
                metadata.append(word)
                if word != '%PADDING%':
                    sentence += ' ' + word
            if step == 0 and i == 0:
                all_emb = c_emb.data.cpu()[:context_lengths[i], i, :]
            else:
                all_emb = torch.cat([all_emb, c_emb.data.cpu()[:context_lengths[i], i, :]], 0)
            metadata = ['_'.join([word, sentence]) for word in metadata]
            all_metadata.extend(metadata)


    G = att_layer(c_emb, q_emb, context_lengths, question_lengths)
    M = model_layer(model_out_h_0, G, context_lengths, step)
    if config['not_pretrain']:
        M_trans = M
        G_trans = G
    else:
        M_trans, G_trans = ner_hw_layer(M, G)
    # M_trans, G_trans = ner_hw_layer(M, G)
    prob = ner_out_layer(M_trans, G_trans, context_lengths)
    # prob = ner_out_layer(M, G, context_lengths)
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
    lst_decode = crf(prob.transpose(0, 1).contiguous(), crf_mask, context_lengths)
    # value, rec_label = torch.max(prob.data, 2)
    if summary_emb:
        return lst_decode, all_emb, all_metadata, q_emb[:, 0, :].data.cpu()
    else:
        return lst_decode


class BoundaryPerformance(object):
    def __init__(self, vocab):
        self._vocab = vocab
        self.reset()

    def reset(self):
        self._hit_num = 0
        self._rec_num = 0
        self._lab_num = 0
        self._unmatch = 0

    def _extract_mentions(self, tokens):
        mentions = []
        pos = 0
        mention_stack = []
        for token in tokens:
            if token == 'X':
                pos += 1
            elif token.startswith('('):
                mention_stack.append((token[1:], pos))
            elif token.startswith(')'):
                mention_type = token[1:]
                is_match = False
                for pre in mention_stack[::-1]:
                    if pre[0] == mention_type:
                        is_match = True
                        mention_stack.remove(pre)
                        mentions.append('_'.join((pre[0], str(pre[1]), str(pos))))
                        break
                if not is_match:
                    self._unmatch += 1
        self._unmatch += len(mention_stack)

        return set(mentions)

    def extract(self, label_list):
        start = 0
        end = 0
        mentions = []
        pre = 0
        mid = label_list[0]
        label_list.append(0)
        for i, back in enumerate(label_list[1:]):
            i += 1
            if i > 1:
                pre = label_list[i-2]
                mid = label_list[i-1]
            if mid == 0:
                continue
            elif pre == 0 and back == 0:
                start = i - 1
                end = i
                mentions.append('_'.join((str(start), str(end))))
            elif pre == 0 and back == 1:
                start = i - 1
            elif pre == 1 and back == 0:
                end = i
                mentions.append('_'.join((str(start), str(end))))
        return set(mentions)




    def evaluate(self, i, label, rec, out_stream=None, pr=True):
        mention_lab = self.extract(label)
        mention_rec = self.extract(rec)


        # label = [self._vocab.getWord(l) for l in label]
        # rec = [self._vocab.getWord(r) for r in rec]
        # if out_stream is not None:
        #     label_str = ' '.join(label)
        #     rec_str = ' '.join(rec)
        #     out_stream.write('{}|||{}\n'.format(label_str, rec_str))
        #     out_stream.flush()
        # mention_lab = self._extract_mentions(label)
        # mention_rec = self._extract_mentions(rec)
        if pr:
            print(i, mention_rec)
        mention_hit = mention_lab.intersection(mention_rec)
        self._lab_num += len(mention_lab)
        self._rec_num += len(mention_rec)
        self._hit_num += len(mention_hit)

    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print('label={}, rec={}, hit={}, unmatch_start={}'.format(self._lab_num, self._rec_num, self._hit_num, self._unmatch))
        print('p={},r={}, f={}'.format(p, r, f))
        return (f, p, r)

def remove_end_tag(tag_lst):
    new_tag = []
    for tag in tag_lst:
        if tag == 0:
            break
        else:
            new_tag.append(tag)
    return new_tag

def evaluate_all(my_arg, pr=True):
    emb = LoadEmbedding('res/emb.txt')
    if config['label_emb'] or config['question_alone']:
        onto_emb = LoadEmbedding('res/onto_embedding.txt')
    print('finish loading embedding')
    # batch_getter = BatchGetter('data/dev', 'GPE_NAM', 1, False)
    batch_getter_lst = []
    if config['bioes']:
        if config['data'] == 'conll':
            pernam_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testa', 'PER', 1, False)
            batch_getter_lst.append(pernam_batch_getter)

            loc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testa', 'LOC', 1, False)
            batch_getter_lst.append(loc_batch_getter)
            if not config['drop_misc']:
                misc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testa', 'MISC', 1, False)
                batch_getter_lst.append(misc_batch_getter)
            #
            org_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testa', 'ORG', 1, False)
            batch_getter_lst.append(org_batch_getter)
        elif config['data'] == 'OntoNotes':
            onto_notes_data = OntoNotesGetter('data/OntoNotes/leaf_test.json', utils.get_ontoNotes_type_lst(), 1, False)
            batch_getter_lst.append(onto_notes_data)


    else:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'PER', 1, False)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'LOC', 1, False)
        batch_getter_lst.append(loc_batch_getter)
        if not config['drop_misc']:
            misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'MISC', 1, False)
            batch_getter_lst.append(misc_batch_getter)
        #
        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'ORG', 1, False)
        batch_getter_lst.append(org_batch_getter)
    batch_size = 100
    batch_getter = MergeBatchGetter(batch_getter_lst, batch_size, False, data_name=config['data'])
    print('finish loading dev data')
    # if config['data'] == 'OntoNotes':
    #     emb_onto = True
    # else:
    #     emb_onto = False
    embedding_layer = EmbeddingLayer(emb, 0)
    if config['label_emb']:
        q_word_embedding = nn.Embedding(onto_emb.get_voc_size(), onto_emb.get_emb_size())
        q_word_embedding.weight.data.copy_(onto_emb.get_embedding_tensor())
        q_word_embedding.weight.requires_grad = False
    else:
        q_word_embedding = None
    d = config['hidden_size']
    if config['question_alone']:
        q_emb_layer = QLabel(onto_emb, 0)
    else:
        q_emb_layer = None
    att_layer = AttentionFlowLayer(2 * d)
    model_layer = ModelingLayer(8 * d, d, 2, 0)
    ner_hw_layer = NerHighway(2 * d, 8 * d, 1)
    ner_out_layer = NerOutLayer(10 * d, len(config['Tags']), 0)
    crf = CRF(config, config['Tags'], len(config['Tags']))
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
    model_dir = 'ner_model' + str(my_arg)

    embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    att_layer.load_state_dict(torch.load(model_dir+'/att_layer.pkl'))
    model_layer.load_state_dict(torch.load(model_dir+'/model_layer.pkl'))
    ner_hw_layer.load_state_dict(torch.load(model_dir+'/ner_hw_layer.pkl'))
    ner_out_layer.load_state_dict(torch.load(model_dir + '/ner_out_layer.pkl'))
    crf.load_state_dict(torch.load(model_dir+'/crf.pkl'))
    if config['question_alone']:
        q_emb_layer.load_state_dict(
            torch.load(model_dir + '/q_emb_layer.pkl', map_location=lambda storage, loc: storage))
    else:
        q_emb_layer = None
    if config['question_alone']:
        q_emb_layer.eval()
    embedding_layer.eval()
    att_layer.eval()
    model_layer.eval()
    ner_hw_layer.eval()
    ner_out_layer.eval()
    crf.eval()

    ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    evaluator = ConllBoundaryPerformance(ner_tag)
    evaluator.reset()
    out_file = codecs.open('data/ner_eva_result'+str(my_arg), mode='wb', encoding='utf-8')

    ex_iterations = 0
    for iteration, this_batch in enumerate(batch_getter):
        top_path = evaluate_one(ex_iterations + iteration, embedding_layer, q_word_embedding, q_emb_layer, att_layer,
                                   model_layer, ner_hw_layer, ner_out_layer, crf, this_batch)
        for batch_no, path in enumerate(top_path):
            evaluator.evaluate(iteration*batch_size+batch_no, remove_end_tag(this_batch[1].numpy()[batch_no, :].tolist()), path, out_file, pr)
        if (iteration+1)*batch_size % 100 == 0:
            print('{} sentences processed'.format((iteration+1)*batch_size))
            evaluator.get_performance()
    return evaluator.get_performance()


if __name__ == '__main__':
    def free_evaluate_all(my_arg, pr=True):
        emb = LoadEmbedding('res/emb.txt')
        if config['label_emb'] or config['question_alone']:
            onto_emb = LoadEmbedding('res/onto_embedding.txt')
        print('finish loading embedding')
        batch_getter_lst = []
        if my_arg == 0:
            # pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'PER', 1, False)
            # batch_getter_lst.append(pernam_batch_getter)

            # loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'LOC', 1, False)
            # batch_getter_lst.append(loc_batch_getter)

            misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'MISC', 1, False)
            batch_getter_lst.append(misc_batch_getter)

            # org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'ORG', 1, False)
            # batch_getter_lst.append(org_batch_getter)

        if my_arg == 1:
            # pernam_batch_getter = ConllBatchGetter('data/ttt', 'PER', 1, False)
            # batch_getter_lst.append(pernam_batch_getter)
            # pernam_batch_getter = ConllBatchGetter('data/ttt', 'singer', 1, False)
            # batch_getter_lst.append(pernam_batch_getter)
            pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testb', 'PER', 1, False)
            batch_getter_lst.append(pernam_batch_getter)

            loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testb', 'LOC', 1, False)
            batch_getter_lst.append(loc_batch_getter)
            #
            misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testb', 'MISC', 1, False)
            batch_getter_lst.append(misc_batch_getter)

            org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testb', 'ORG', 1, False)
            batch_getter_lst.append(org_batch_getter)
        if my_arg == 2:
            # pernam_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testb', 'food', 1, False)
            # batch_getter_lst.append(pernam_batch_getter)
            pernam_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testb', 'PER', 1, False)
            batch_getter_lst.append(pernam_batch_getter)

            loc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testb', 'LOC', 1, False)
            batch_getter_lst.append(loc_batch_getter)

            misc_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testb', 'MISC', 1, False)
            batch_getter_lst.append(misc_batch_getter)

            org_batch_getter = ConllBatchGetter('data/conll2003/bioes_eng.testb', 'ORG', 1, False)
            batch_getter_lst.append(org_batch_getter)
        if my_arg == 3:
            # onto_notes = OntoNotesGetter('data/OntoNotes/test.json', '/person', 1, False)
            # batch_getter_lst.append(onto_notes)
            onto_notes_data = OntoNotesGetter('data/OntoNotes/test.json', utils.get_ontoNotes_type_lst(), 1, False)
            batch_getter_lst.append(onto_notes_data)
        batch_size = 100
        batch_getter = MergeBatchGetter(batch_getter_lst, batch_size, False, data_name=config['data'])
        print('finish loading dev data')
        # if config['data'] == 'OntoNotes':
        #     emb_onto = True
        # else:
        #     emb_onto = False
        embedding_layer = EmbeddingLayer(emb, 0)
        if config['label_emb']:
            q_word_embedding = nn.Embedding(onto_emb.get_voc_size(), onto_emb.get_emb_size())
            q_word_embedding.weight.data.copy_(onto_emb.get_embedding_tensor())
            q_word_embedding.weight.requires_grad = False
        else:
            q_word_embedding = None
        d = config['hidden_size']
        if config['question_alone']:
            q_emb_layer = QLabel(onto_emb, 0)
        else:
            q_emb_layer = None
        att_layer = AttentionFlowLayer(2 * d)
        model_layer = ModelingLayer(8 * d, d, 2, 0)
        ner_hw_layer = NerHighway(2 * d, 8 * d, 1)
        ner_out_layer = NerOutLayer(10 * d, len(config['Tags']), 0)
        crf = CRF(config, config['Tags'], len(config['Tags']))
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
        model_dir = 'ner_model8'

        att_layer.load_state_dict(torch.load(model_dir + '/early_att_layer.pkl', map_location=lambda storage, loc: storage))
        model_layer.load_state_dict(torch.load(model_dir + '/early_model_layer.pkl', map_location=lambda storage, loc: storage))
        ner_hw_layer.load_state_dict(torch.load(model_dir + '/early_ner_hw_layer.pkl', map_location=lambda storage, loc: storage))
        ner_out_layer.load_state_dict(torch.load(model_dir + '/early_ner_out_layer.pkl', map_location=lambda storage, loc: storage))
        crf.load_state_dict(torch.load(model_dir + '/early_crf.pkl', map_location=lambda storage, loc: storage))
        embedding_layer.load_state_dict(torch.load(model_dir + '/early_embedding_layer.pkl', map_location=lambda storage, loc: storage))
        if config['question_alone']:
            q_emb_layer.load_state_dict(
                torch.load(model_dir + '/q_emb_layer.pkl', map_location=lambda storage, loc: storage))
        else:
            q_emb_layer = None
        if config['question_alone']:
            q_emb_layer.eval()
        embedding_layer.eval()
        att_layer.eval()
        model_layer.eval()
        ner_hw_layer.eval()
        ner_out_layer.eval()
        crf.eval()

        ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
        if my_arg == 3:
            evaluator = ConllBoundaryPerformance(ner_tag, onto_notes_data)
        else:
            evaluator = ConllBoundaryPerformance(ner_tag)
        evaluator.reset()
        out_file = codecs.open('data/eva_result' + str(my_arg), mode='wb', encoding='utf-8')
        # writer.add_embedding(embedding_layer.word_embedding.weight.data.cpu())
        # return
        all_emb = None
        all_metadata = []
        ex_iterations = 0
        summary_emb = False
        for iteration, this_batch in enumerate(batch_getter):
            # if iteration >= 15:
            #     break

            if summary_emb:
                top_path, all_emb, all_metadata, q = evaluate_one(ex_iterations + iteration, embedding_layer, q_word_embedding, q_emb_layer, att_layer,
                                        model_layer, ner_hw_layer, ner_out_layer, crf, this_batch, summary_emb, all_emb, all_metadata)
            else:
                top_path = evaluate_one(ex_iterations + iteration, embedding_layer, q_word_embedding, q_emb_layer, att_layer,
                                        model_layer, ner_hw_layer, ner_out_layer, crf, this_batch)
            for batch_no, path in enumerate(top_path):
                evaluator.evaluate(iteration*batch_size+batch_no, remove_end_tag(this_batch[1].numpy()[batch_no, :].tolist()), path,
                                   out_file, pr)
            if (iteration + 1) * batch_size % 100 == 0:
                print('{} sentences processed'.format((iteration + 1) * batch_size))
                evaluator.get_performance()
        if summary_emb:
            writer.add_embedding(torch.cat([q, all_emb], 0), metadata=['question'] + all_metadata)
        return evaluator.get_performance()
    config['cuda_num'] = 1
    torch.cuda.set_device(config['cuda_num'])
    config['use_gaz'] = False
    # config['use_gaz'] = False
    # config['char_emb_dim'] = 8
    # config['question_alone'] = True
    # config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4}
    config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4, 'E': 5, 'S': 6}
    config['misc'] = True
    config['bioes'] = True
    config['sigmoid'] = False
    config['entity_emb'] = False
    config['gate'] = False
    config['use_dropout'] = False
    config['large_crf'] = True
    config['data'] = 'conll'
    config['question_alone'] = False
    config['label_emb'] = False
    config['not_pretrain'] = False
    # config['gate'] = True
    # writer.add_embedding(torch.rand(4, 10))
    # print 'ddd'
    free_evaluate_all(2, False)
