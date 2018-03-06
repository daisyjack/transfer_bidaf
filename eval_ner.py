import torch
from configurations import config, to_np, Vocab
from batch_getter import BatchGetter, get_question, MergeBatchGetter
import time
from bidaf import LoadEmbedding, EmbeddingLayer, AttentionFlowLayer, ModelingOutLayer

from torch.autograd import Variable
import numpy
from masked_cross_entropy import masked_cross_entropy
from torch.nn.utils import clip_grad_norm
import codecs
from conll_data_trans import ConllBatchGetter, ConllBoundaryPerformance
torch.manual_seed(0)





def evaluate_one(step, embedding_layer, att_layer, model_out_layer, emb_opt, att_opt, model_out_opt, this_batch):
    emb_opt.zero_grad()
    att_opt.zero_grad()
    model_out_opt.zero_grad()

    d = embedding_layer.out_dim
    this_batch_num = len(this_batch[2])

    # question = Variable(get_question('%PER%', this_batch_num), volatile=True)  # (batch, J=1, 51)
    question = Variable(this_batch[4])
    question_lengths = [1 for _ in range(this_batch_num)]
    context = Variable(this_batch[0], volatile=True)  # (batch, T, 51)
    context_lengths = this_batch[3]  # list
    target = Variable(this_batch[1], volatile=True)  # (batch, T)
    emb_h_0 = Variable(torch.zeros(2, this_batch_num, d), volatile=True)
    model_out_h_0 = Variable(torch.zeros(2*model_out_layer.num_layers, this_batch_num, d), volatile=True)
    con_lens_var = Variable(torch.LongTensor(context_lengths), volatile=True)

    if config['USE_CUDA']:
        question = question.cuda(config['cuda_num'])
        context = context.cuda(config['cuda_num'])
        target = target.cuda(config['cuda_num'])
        emb_h_0 = emb_h_0.cuda(config['cuda_num'])
        model_out_h_0 = model_out_h_0.cuda(config['cuda_num'])
        con_lens_var = con_lens_var.cuda(config['cuda_num'])

    c_emb = embedding_layer(context, emb_h_0, context_lengths)
    q_emb = embedding_layer(question, emb_h_0, question_lengths)
    G = att_layer(c_emb, q_emb, context_lengths, question_lengths)
    prob = model_out_layer(model_out_h_0, G, context_lengths)
    value, rec_label = torch.max(prob.data, 2)
    return target.cpu().data.squeeze(0), rec_label.cpu().squeeze(0)


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
            print i, mention_rec
        mention_hit = mention_lab.intersection(mention_rec)
        self._lab_num += len(mention_lab)
        self._rec_num += len(mention_rec)
        self._hit_num += len(mention_hit)

    def get_performance(self):
        p = float(self._hit_num) / float(self._rec_num) if self._rec_num > 0 else 0.0
        r = float(self._hit_num) / float(self._lab_num) if self._lab_num > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        f = f * 100
        print 'label={}, rec={}, hit={}, unmatch_start={}'.format(self._lab_num, self._rec_num, self._hit_num, self._unmatch)
        print 'p={},r={}, f={}'.format(p, r, f)
        return (f, p, r)



def evaluate_all(my_arg, pr=True):
    emb = LoadEmbedding('res/emb.txt')
    print 'finish loading embedding'
    # batch_getter = BatchGetter('data/dev', 'GPE_NAM', 1, False)
    batch_getter_lst = []
    if my_arg == 0:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'PER', 1, False)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'LOC', 1, False)
        batch_getter_lst.append(loc_batch_getter)

        misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'MISC', 1, False)
        batch_getter_lst.append(misc_batch_getter)

        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'ORG', 1, False)
        batch_getter_lst.append(org_batch_getter)

    if my_arg == 1:
        pernam_batch_getter = BatchGetter('data/dev', 'PER_NAM', 1, False)
        batch_getter_lst.append(pernam_batch_getter)

        fac_batch_getter = BatchGetter('data/dev', 'FAC_NAM', 1, False)
        batch_getter_lst.append(fac_batch_getter)

        loc_batch_getter = BatchGetter('data/dev', 'LOC_NAM', 1, False)
        batch_getter_lst.append(loc_batch_getter)

        gpe_batch_getter = BatchGetter('data/dev', 'GPE_NAM', 1, False)
        batch_getter_lst.append(gpe_batch_getter)

        org_batch_getter = BatchGetter('data/dev', 'ORG_NAM', 1, False)
        batch_getter_lst.append(org_batch_getter)
    if my_arg == 2:
        pernam_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'PER', 1, False)
        batch_getter_lst.append(pernam_batch_getter)

        loc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'LOC', 1, False)
        batch_getter_lst.append(loc_batch_getter)

        misc_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'MISC', 1, False)
        batch_getter_lst.append(misc_batch_getter)

        org_batch_getter = ConllBatchGetter('data/conll2003/bio_eng.testa', 'ORG', 1, False)
        batch_getter_lst.append(org_batch_getter)

    batch_getter = MergeBatchGetter(batch_getter_lst, 1, False)
    print 'finish loading dev data'
    embedding_layer = EmbeddingLayer(emb, 0)
    d = embedding_layer.get_out_dim()
    att_layer = AttentionFlowLayer(2 * d)
    # if my_arg == 2:
    model_out_layer = ModelingOutLayer(8*d, d, 2, 3, 0)
    # else:
    #     model_out_layer = ModelingOutLayer(8*d, d, 2, 2, 0)
    model_dir = 'model' + str(my_arg)

    embedding_layer.load_state_dict(torch.load(model_dir+'/embedding_layer.pkl'))
    att_layer.load_state_dict(torch.load(model_dir+'/att_layer.pkl'))
    model_out_layer.load_state_dict(torch.load(model_dir+'/model_out_layer.pkl'))

    # models = [embedding_layer, att_layer, model_out_layer]
    # opts = [emb_opt, att_opt, model_out_opt]
    ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # if my_arg == 2:
    evaluator = ConllBoundaryPerformance(ner_tag)
    # else:
    #     evaluator = BoundaryPerformance(ner_tag)
    evaluator.reset()


    if config['USE_CUDA']:
        att_layer.cuda(config['cuda_num'])
        embedding_layer.cuda(config['cuda_num'])
        model_out_layer.cuda(config['cuda_num'])

    emb_opt = torch.optim.Adam(embedding_layer.parameters())
    att_opt = torch.optim.Adam(att_layer.parameters())
    model_out_opt = torch.optim.Adam(model_out_layer.parameters())
    out_file = codecs.open('data/eva_result'+str(my_arg), mode='wb', encoding='utf-8')

    ex_iterations = 0
    for iteration, this_batch in enumerate(batch_getter):
        target, rec = evaluate_one(ex_iterations + iteration, embedding_layer, att_layer,
                                   model_out_layer, emb_opt, att_opt, model_out_opt, this_batch)
        evaluator.evaluate(iteration, target.numpy().tolist(), rec.numpy().tolist(), out_file, pr=pr)
        if iteration % 100 == 0:
            print '{} sentences processed'.format(iteration)
            evaluator.get_performance()
    return evaluator.get_performance()


if __name__ == '__main__':
    # print get_question(3)
    # ner_tag = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    # evaluator = BoundaryPerformance(ner_tag)
    # evaluator.reset()
    # out_file = codecs.open('data/eva_result.txt', mode='wb', encoding='utf-8')
    # evaluator.evaluate(1, [0,1,1,0,1,1,0,1], [0,1,1,0,1,1,0,0], out_file)
    evaluate_all(1)
    # print get_question(1)

