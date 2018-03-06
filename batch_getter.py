# coding: utf-8

import codecs
import random
from configurations import config
import torch
import os
import numpy
import utils

def pad_tgt_seq(seq, max_length):
    # max_length = 350
    pad = numpy.zeros((max_length - seq.size), dtype='int32')
    pad_seq = numpy.hstack((seq, pad))
    return pad_seq

def pad_src_seq(seq, max_length):
    # max_length = 350
    pad = numpy.zeros((max_length-seq.shape[0], seq.shape[1]), dtype='int32')
    pad_seq = numpy.vstack((seq, pad))
    return pad_seq




def get_source_mask(batch, hidden_size, max_length, lengths):
    mask = torch.zeros(batch, max_length, hidden_size)
    for i in range(batch):
        if lengths[i] > 0:
            mask[i, :lengths[i], :] = 1
    return mask.transpose(0, 1)

def get_target_mask(batch, max_length, lengths):
    mask = torch.zeros(batch, max_length)
    for i in range(batch):
        mask[i, :lengths[i]] = 1
    return mask.transpose(0, 1)


class MergeBatchGetter(object):
    def __init__(self, batch_getter_lst, batch_size, shuffle=True, data_name='conll'):
        self.data_name = data_name
        if self.data_name == 'OntoNotes':
            self.feaMats = batch_getter_lst[0].feaMats
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.cursor = 0
        self.all_samples = []
        for batch_getter in batch_getter_lst:
            self.all_samples.extend(batch_getter.all_samples)
            batch_getter.all_samples = []
        self.sample_num = len(self.all_samples)
        self.gazetteers = []

        if config['use_gaz']:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']

            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self.gazetteers.append(load_gaz_list(gazfile))

        self.reset()

    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def __next__(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            if self.data_name == 'OntoNotes':
                edit_required_batch = []
                for item in required_batch:
                    edit_required_batch.append((self.feaMats[item[0]], item[1], item[2]))
                required_batch = edit_required_batch

            self.cursor += self.batch_size
            input_seqs = [seq_label[0] for seq_label in required_batch]
            input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 350
            # input_labels_length[-1] = 350


            questions = [get_question(seq_label[2], 1, self.gazetteers) for seq_label in required_batch]
            questions_len = [s.shape[0] for s in questions]
            que_padded = [pad_src_seq(s, max(questions_len))[numpy.newaxis, ...] for s in questions]
            que_tensor = torch.from_numpy(numpy.concatenate(que_padded, axis=0)).type(torch.LongTensor)

            return seq_tensor, label_tensor, input_labels_length, input_seqs_length, que_tensor, questions_len
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)

        self.cursor = 0





class BatchGetter(object):
    def __init__(self, file_name, require_type, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.require_type = require_type
        self._vocabs = config['Vocabs']
        self._outtag_voc = config['OutTags']
        self._fea_pos = config['fea_pos']
        self._word_pos = config['WordPos']

        self._vocab_char = config['CharVoc']
        self._max_char_len = config['max_char']

        self._use_char_conv = config['use_char_conv']

        self._use_gaz = config['use_gaz']


        if config['use_gaz']:
            gazdir = config['GazetteerDir']
            gaz_names = config['Gazetteers']
            self._gazetteers = []
            for (id, gaz) in enumerate(gaz_names):
                gazfile = os.path.join(gazdir, gaz)
                self._gazetteers.append(self._load_gaz_list(gazfile))

        # 游标
        self.cursor = 0

        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        all_samples = []

        # self.all_samples: [(tokens, labels),()]
        fea_len = len(self._vocabs)
        if self._use_gaz:
            fea_len += len(self._gazetteers)
        if self._use_char_conv:
            fea_len += self._max_char_len * 2



        for line in train_file:
            line = line.strip()
            if line:
                parts = line.split('|||')
                src_tokens = parts[0].strip().split(' ')
                tgt_tokens = parts[1].strip().split(' ')
                feaMat = numpy.zeros((len(src_tokens), fea_len),
                                     dtype='int32')


                for (lid, token) in enumerate(src_tokens):
                    parts = token.split('#')
                    for (i, voc) in enumerate(self._vocabs):
                        fpos = self._fea_pos[i]
                        wid = voc.getID(parts[fpos])
                        feaMat[lid, i] = wid
                    curr_end = len(self._vocabs)
                    if self._use_gaz:
                        gazStart = len(self._vocabs)
                        for (id, gaz) in enumerate(self._gazetteers):
                            if parts[0] in gaz:
                                feaMat[lid, id + gazStart] = 1
                        curr_end += len(self._gazetteers)
                    if self._use_char_conv:
                        word = parts[self._word_pos]
                        chStart = curr_end
                        chMaskStart = chStart + self._max_char_len
                        for i in range(len(word)):
                            if i >= self._max_char_len:
                                break
                            feaMat[lid, chStart + i] = self._vocab_char.getID(word[i])
                            feaMat[lid, chMaskStart + i] = 1
                num = len(tgt_tokens) - 1
                # for (lid, token) in enumerate(tgt_tokens):
                #     # if lid != num:
                #     label[lid] = self._outtag_voc.getID(token)
                mentions = self._extract_mentions(tgt_tokens)
                X_num = tgt_tokens.count('X')
                label = numpy.zeros((X_num,), dtype='int32')
                for mention in mentions:
                    mention_type, nam_nom, start, end = mention.split('_')
                    mention_type = mention_type + '_' + nam_nom
                    if mention_type == self.require_type:
                        label[int(start):int(end)] = 1
                question = self.require_type

                # if self.require_type == 'PER_NAM':
                #     question = '%PER_NAM%'
                # elif self.require_type == 'PER_NOM':
                #     question = 'PER_NOM'
                # elif self == 'GPE_NAM':
                #     question = '%GPE%'
                # elif self.require_type.startswith('LOC'):
                #     question = '%LOC%'
                # elif self.require_type.startswith('FAC'):
                #     question = '%FAC%'
                # elif self.require_type.startswith('ORG'):
                #     question = '%ORG%'

                all_samples.append((feaMat, label, question))
        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def _load_gaz_list(self, file):
        words=set()
        with open(file) as f:
            for line in f:
                words.add(line.strip())
        return words

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


    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def next(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]
            # required_batch = self.all_samples[:config['batch_size']]
            self.cursor += self.batch_size
            input_seqs = [seq_label[0] for seq_label in required_batch]
            input_labels = [seq_label[1] for seq_label in required_batch]
            input_seqs_length = [s.shape[0] for s in input_seqs]
            input_labels_length = [s.size for s in input_labels]
            seqs_padded = [pad_src_seq(s, max(input_seqs_length))[numpy.newaxis, ...] for s in input_seqs]
            labels_padded = [pad_tgt_seq(s, max(input_labels_length))[numpy.newaxis, ...] for s in input_labels]
            # (batch, max_seq, len(embnames)+len(gazs)+max_char+max_char)
            seq_tensor = torch.from_numpy(numpy.concatenate(seqs_padded, axis=0)).type(torch.LongTensor)
            # (batch, max_label)
            label_tensor = torch.from_numpy(numpy.concatenate(labels_padded, axis=0)).type(torch.LongTensor)

            # input_seqs_length[-1] = 350
            # input_labels_length[-1] = 350


            return seq_tensor, label_tensor, input_labels_length, input_seqs_length
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)

        self.cursor = 0

def load_gaz_list(file):
    words=set()
    with codecs.open(file, mode='rb', encoding='utf-8') as f:
        for line in f:
            words.add(line.strip())
    return words


def get_question(question, batch_num, gazetteers):
    max_char_len = config['max_char']
    if question == 'PER_NAM':
        question = 'who'# is a person'
    elif question == 'PER_NOM':
        question = 'who is the nominal person'
    elif question == 'GPE_NAM':
        question = 'which'# is geopolitical'
    elif question == 'GPE_NON':
        question = 'which is the nominal geopolitical entity'
    elif question == 'LOC_NAM':
        question = 'where'# is a location'
    elif question == 'LOC_NOM':
        question = 'where is the nominal location'
    elif question == 'FAC_NAM':
        question = 'what'# is a facility'
    elif question == 'FAC_NOM':
        question = 'what is the nominal facility'
    elif question == 'ORG_NAM':
        question = 'what'# is an organization'
    elif question == 'ORG_NOM':
        question = 'what is the nominal organization'
    elif question == 'PER':
        if config['entity_emb']:
            question = '%PER%'
        else:
            question = 'who'# is a person'
    elif question == 'LOC':
        if config['entity_emb']:
            question = '%LOC%'
        else:
            question = 'where'# is a location'
    elif question == 'ORG':
        if config['entity_emb']:
            question = '%ORG%'
        else:
            question = 'organization'#'what is an organization'
    elif question == 'MISC':
        if config['entity_emb']:
            question = '%MISC%'
        else:
            if config['misc']:
                question = 'which'
            else:
                question = 'entity'
    elif question == 'food':
        question = 'food'
    elif question == 'singer':
        question = 'singer'
    elif question == '/person':
        question = 'who'
    elif question[0] == '/':
        # question = ' '.join(question.split('/')[1:])
        pass
    # question = '%PER% person who man woman'
    words = question.split(' ')
    # if config['use_gaz']:
    #     gazdir = config['GazetteerDir']
    #     gaz_names = config['Gazetteers']
    #     gazetteers = []
    #     for (id, gaz) in enumerate(gaz_names):
    #         gazfile = os.path.join(gazdir, gaz)
    #         gazetteers.append(load_gaz_list(gazfile))
    # max_char_len = config['max_char']
    # if config['use_gaz']:
    #     fea_len = 57
    # else:
    #     fea_len = 51
    #
    # feaMat = numpy.zeros((len(words), fea_len), dtype='int32')
    # for word_no, word in enumerate(words):
    #     feaMat[word_no, 0] = config['Vocabs'][0].getID(word)
    #     curr_end = len(config['Vocabs'])
    #     if config['use_gaz']:
    #         gazStart = curr_end
    #         for (id, gaz) in enumerate(gazetteers):
    #             if word.lower() in gaz:
    #                 feaMat[word_no, id + gazStart] = 1
    #         curr_end += len(gazetteers)
    #     if config['use_char_conv']:
    #         chStart = curr_end
    #         chMaskStart = chStart + max_char_len
    #         for i in range(len(word)):
    #             if i >= max_char_len:
    #                 break
    #             feaMat[word_no, chStart + i] = config['CharVoc'].getID(word[i])
    #             feaMat[word_no, chMaskStart + i] = 1
    feaMat = utils.get_context_mat(words, config)
    # batch = [feaMat[numpy.newaxis, ...] for _ in range(batch_num)]
    # return torch.from_numpy(numpy.concatenate(batch, axis=0)).type(torch.LongTensor)
    return feaMat  # (S, 51)



# class Batch_gen(object):
#     def __init__(self, file_name, batch_size):
#         self.batch_size = batch_size
#         self.file = codecs.open(file_name, 'rb')
#         self.pairs = []
#         for line in self.file:
#             if line.strip():
#                 str_pair = line.strip().split('\t')
#                 str_seq = str_pair[0].split(' ')
#                 seq = []
#                 for i in str_seq:
#                     seq.append(int(i))
#                 int_pair = [seq, int(str_pair[1])]
#                 self.pairs.append(int_pair)
#         self.cursor = 0
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         if self.cursor < len(self.pairs):
#             result = self.pairs[self.cursor:self.cursor+self.batch_size]
#             self.cursor += self.batch_size
#             seq_pad = []
#             label_pad
#             biggest = 0
#             for pair in result:
#                 length = len(pair[0])
#                 lengths.append(length)
#                 if length > biggest:
#                     biggest = length
#             for pair in result:
#                 if len(pair)
#
#             return result
#         else:
#             raise StopIteration("out of list")
# data = Batch_gen('data', 4)
# for i, batch in enumerate(data):
#     print i, batch
if __name__ == "__main__":
    pernam_batch_getter = BatchGetter('data/train.txt', 'PER_NAM', 1,False)
    pernom_batch_getter = BatchGetter('data/train.txt', 'PER_NOM',1)
    fac_batch_getter = BatchGetter('data/train.txt', 'FAC_NAM',1)
    loc_batch_getter = BatchGetter('data/train.txt', 'LOC_NAM',1)
    gpe_batch_getter = BatchGetter('data/train.txt', 'GPE_NAM',1)
    org_batch_getter = BatchGetter('data/train.txt', 'ORG_NAM',1)
    b = MergeBatchGetter([pernam_batch_getter, pernom_batch_getter, fac_batch_getter, loc_batch_getter,
                          gpe_batch_getter, org_batch_getter], 8, False)
    c = next(b)
    # print batch_getter._unmatch
    pass
    # print get_source_mask(3, 4, [4,6,5]).transpose(0,1)