# coding: utf-8
from __future__ import print_function
import torch
import codecs
from configurations import config, fg_config
from batch_getter import pad_src_seq, pad_tgt_seq, MergeBatchGetter
import numpy
import random
import os
import json
import utils

class OntoNotesGetter:
    def __init__(self, file_name, require_type_lst, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        # self.require_type = require_type
        self.sentences = []
        self.feaMats = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        blank_line = 0
        all_samples = []
        src_tokens = []
        targets = []


        for line_no, line in enumerate(train_file):
            line = json.loads(line)
            src_tokens = line['tokens']
            mentions = line['mentions']
            feaMat = utils.get_context_mat(src_tokens, config)
            self.feaMats.append(feaMat)

            for require_type in require_type_lst:
                label = numpy.zeros((len(src_tokens),), dtype='int32')
                label.fill(config['Tags']['O'])

                has_this_require_type = False

                for mention in mentions:

                    require = False
                    # lengths = [len(label.split('/')) for label in mention['labels']]
                    # if len(lengths) == 0:
                    #     continue
                    # label_leaf = max(lengths)
                    # new_labels = [label for label in mention['labels'] if len(label.split('/')) == label_leaf]
                    if require_type in mention['labels']:
                        require = True
                        has_this_require_type = True
                    start = mention['start']
                    end = mention['end']
                    if require:

                        if end-start == 1:
                            label[start] = config['Tags']['S']
                        elif end-start == 2:
                            label[start] = config['Tags']['B']
                            label[start+1] = config['Tags']['E']
                        elif end-start >= 3:
                            label[start] = config['Tags']['B']
                            label[start+1:end-1] = config['Tags']['I']
                            label[end-1] = config['Tags']['E']
                if has_this_require_type:

                    question = require_type

                    all_samples.append((line_no, label, question))

            self.sentences.append(zip(src_tokens, label))

        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)

class TrainOntoNotesGetter:
    def __init__(self, file_name, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.sentences = []
        self.cursor = 0
        self.feaMats = []
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        blank_line = 0
        all_samples = []
        src_tokens = []
        targets = []


        for line_no, line in enumerate(train_file ):
            line = json.loads(line)
            src_tokens = line['tokens']
            mentions = line['mentions']
            feaMat = utils.get_context_mat(src_tokens, config)
            self.feaMats.append(feaMat)


            # this_mention_types = set([])
            # for mention in mentions:
            #     this_mention_types.update(mention['labels'])
            this_mention_types = set([])
            new_mentions_labels = []
            for mention in mentions:
                lengths = [len(label.split('/')) for label in mention['labels']]
                label_leaf = max(lengths)
                new_labels = [label for label in mention['labels'] if len(label.split('/'))==label_leaf]
                new_mentions_labels.append(new_labels)
            for a in new_mentions_labels:
                this_mention_types.update(a)


            for require_type in this_mention_types:
                label = numpy.zeros((len(src_tokens),), dtype='int32')
                label.fill(config['Tags']['O'])
                for i, mention in enumerate(mentions):

                    require = False
                    if require_type in new_mentions_labels[i]:
                        require = True
                    start = mention['start']
                    end = mention['end']
                    if require:

                        if end-start == 1:
                            label[start] = config['Tags']['S']
                        elif end-start == 2:
                            label[start] = config['Tags']['B']
                            label[start+1] = config['Tags']['E']
                        elif end-start >= 3:
                            label[start] = config['Tags']['B']
                            label[start+1:end-1] = config['Tags']['I']
                            label[end-1] = config['Tags']['E']

                question = require_type

                all_samples.append((line_no, label, question))

            self.sentences.append(zip(src_tokens, label))

        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)


class OntoNotesNZGetter:
    def __init__(self, file_name, require_type_lst, batch_size, shuffle=True, req_depth=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0
        self._unmatch = 0
        self.sentences = []
        self.feaMats = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        self.all_samples = []


        for line_no, line in enumerate(train_file):
            line = json.loads(line)
            src_tokens = line['tokens']
            mentions = line['mentions']
            # feaMat = utils.get_fg_mat(src_tokens, config)
            # self.feaMats.append(feaMat)

            for mention in mentions:
                is_req = self.check_depth(mention, req_depth)
                if is_req and mention['start'] >= 0:
                    start = mention['start']
                    end = mention['end']
                    men_tokens = src_tokens[start:end]
                    l_ctx_start = start-fg_config['ctx_window_size'] if start-fg_config['ctx_window_size'] >= 0 else 0
                    l_ctx_end = start
                    r_ctx_start = end+1
                    r_ctx_end = r_ctx_start+fg_config['ctx_window_size']
                    l_ctx_tokens = src_tokens[l_ctx_start:l_ctx_end]
                    r_ctx_tokens = src_tokens[r_ctx_start:r_ctx_end]
                    men_mat = utils.get_fg_mat(men_tokens, fg_config)
                    l_ctx_mat = utils.get_ctx_mat(l_ctx_tokens, fg_config)
                    r_ctx_mat = utils.get_ctx_mat(r_ctx_tokens, fg_config)
                    label = numpy.zeros((1, len(require_type_lst)), dtype='int32')
                    for i, require_type in enumerate(require_type_lst):
                        if require_type in mention['labels']:
                            label[0, i] = 1
                    type_mat = utils.get_fg_mat(require_type_lst, fg_config, True)
                    self.all_samples.append([l_ctx_mat, men_mat, r_ctx_mat, len(l_ctx_tokens), len(r_ctx_tokens), label, mention['labels'], type_mat])

        train_file.close()
        # self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def __next__(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]

            self.cursor += self.batch_size
            l_ctx = [sample[0] for sample in required_batch]
            mentions = [sample[1] for sample in required_batch]
            r_ctx = [sample[2] for sample in required_batch]
            labels = [sample[5] for sample in required_batch]
            types_str = [sample[6] for sample in required_batch]
            types = required_batch[0][7]
            # l_ctx_lens = [l.shape[0] for l in l_ctx]
            l_ctx_lens = [sample[3] for sample in required_batch]
            men_lens = [m.shape[0] for m in mentions]
            # r_ctx_lens = [r.shape[0] for r in r_ctx]
            r_ctx_lens = [sample[4] for sample in required_batch]
            l_ctx_padded = [pad_src_seq(l, fg_config['ctx_window_size'])[numpy.newaxis, ...] for l in l_ctx]
            mentions_padded = [pad_src_seq(m, max(men_lens))[numpy.newaxis, ...] for m in mentions]
            r_ctx_padded = [pad_src_seq(r, fg_config['ctx_window_size'])[numpy.newaxis, ...] for r in r_ctx]
            # (B, S, 1)
            l_ctx_tensor = torch.from_numpy(numpy.concatenate(l_ctx_padded, axis=0)).type(torch.LongTensor)
            # (B, S, 1)
            mentions_tensor = torch.from_numpy(numpy.concatenate(mentions_padded, axis=0)).type(torch.LongTensor)
            # (B, S, 1)
            r_ctx_tensor = torch.from_numpy(numpy.concatenate(r_ctx_padded, axis=0)).type(torch.LongTensor)
            # (B, 89)
            labels_tensor = torch.from_numpy(numpy.concatenate(labels, axis=0)).type(torch.FloatTensor)
            # (89, 1)
            types_tensor = torch.from_numpy(types).type(torch.LongTensor)

            return {'l_ctx_tensor': l_ctx_tensor, 'l_ctx_lens': l_ctx_lens, 'mentions_tensor': mentions_tensor, 'men_lens': men_lens,
                    'r_ctx_tensor': r_ctx_tensor, 'r_ctx_lens': r_ctx_lens, 'labels_tensor': labels_tensor, 'types_tensor': types_tensor,
                    'types_str': types_str}
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)

        self.cursor = 0

    def check_depth(self, mention, req_depth):
        if req_depth is None:
            return True
        depths = []
        for label in mention['labels']:
            depths.append(len(label.split('/')) - 1)
        if len(depths) > 0 and max(depths) == req_depth:
            return True
        else:
            return False







class OntoNotesFGGetter:
    def __init__(self, file_name, require_type_lst, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0
        self._unmatch = 0
        self.sentences = []
        self.feaMats = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        self.all_samples = []


        for line_no, line in enumerate(train_file):
            line = json.loads(line)
            src_tokens = line['tokens']
            mentions = line['mentions']
            # feaMat = utils.get_fg_mat(src_tokens, config)
            # self.feaMats.append(feaMat)

            for mention in mentions:
                start = mention['start']
                end = mention['end']
                men_tokens = src_tokens[start:end]
                l_ctx_start = start-fg_config['ctx_window_size'] if start-fg_config['ctx_window_size'] >= 0 else 0
                l_ctx_end = start
                r_ctx_start = end+1
                r_ctx_end = r_ctx_start+fg_config['ctx_window_size']
                l_ctx_tokens = src_tokens[l_ctx_start:l_ctx_end]
                r_ctx_tokens = src_tokens[r_ctx_start:r_ctx_end]
                men_mat = utils.get_fg_mat(men_tokens, fg_config)
                l_ctx_mat = utils.get_ctx_mat(l_ctx_tokens, fg_config)
                r_ctx_mat = utils.get_ctx_mat(r_ctx_tokens, fg_config)
                idx = len(self.feaMats)
                self.feaMats.append([l_ctx_mat, men_mat, r_ctx_mat, len(l_ctx_tokens), len(r_ctx_tokens)])
                for require_type in require_type_lst:
                    label = numpy.zeros((1, 1), dtype='int32')
                    if require_type in mention['labels']:
                        label[0, 0] = 1
                    type_mat = utils.get_fg_mat([require_type], fg_config)
                    self.all_samples.append([idx, label, type_mat, require_type])

        train_file.close()
        # self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def __iter__(self):
        return self

    # 在一个epoch内获得一个batch
    def __next__(self):
        if self.cursor < self.sample_num:
            required_batch = self.all_samples[self.cursor:self.cursor+self.batch_size]

            self.cursor += self.batch_size
            l_ctx = [self.feaMats[sample[0]][0] for sample in required_batch]
            mentions = [self.feaMats[sample[0]][1] for sample in required_batch]
            r_ctx = [self.feaMats[sample[0]][2] for sample in required_batch]
            labels = [sample[1] for sample in required_batch]
            types = [sample[2] for sample in required_batch]
            types_str = [sample[3] for sample in required_batch]
            # l_ctx_lens = [l.shape[0] for l in l_ctx]
            l_ctx_lens = [self.feaMats[sample[0]][3] for sample in required_batch]
            men_lens = [m.shape[0] for m in mentions]
            # r_ctx_lens = [r.shape[0] for r in r_ctx]
            r_ctx_lens = [self.feaMats[sample[0]][4] for sample in required_batch]
            l_ctx_padded = [pad_src_seq(l, fg_config['ctx_window_size'])[numpy.newaxis, ...] for l in l_ctx]
            mentions_padded = [pad_src_seq(m, max(men_lens))[numpy.newaxis, ...] for m in mentions]
            r_ctx_padded = [pad_src_seq(r, fg_config['ctx_window_size'])[numpy.newaxis, ...] for r in r_ctx]
            # (B, S, 1)
            l_ctx_tensor = torch.from_numpy(numpy.concatenate(l_ctx_padded, axis=0)).type(torch.LongTensor)
            # (B, S, 1)
            mentions_tensor = torch.from_numpy(numpy.concatenate(mentions_padded, axis=0)).type(torch.LongTensor)
            # (B, S, 1)
            r_ctx_tensor = torch.from_numpy(numpy.concatenate(r_ctx_padded, axis=0)).type(torch.LongTensor)
            # (B, 1)
            labels_tensor = torch.from_numpy(numpy.concatenate(labels, axis=0)).type(torch.FloatTensor)
            # (B, 1)
            types_tensor = torch.from_numpy(numpy.concatenate(types, axis=0)).type(torch.LongTensor)

            return {'l_ctx_tensor': l_ctx_tensor, 'l_ctx_lens': l_ctx_lens, 'mentions_tensor': mentions_tensor, 'men_lens': men_lens,
                    'r_ctx_tensor': r_ctx_tensor, 'r_ctx_lens': r_ctx_lens, 'labels_tensor': labels_tensor, 'types_tensor': types_tensor,
                    'types_str': types_str}
        else:
            raise StopIteration("out of list")

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.all_samples)

        self.cursor = 0




class ConllBatchGetter(object):
    def __init__(self, file_name, require_type, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.require_type = require_type
        self.sentences = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        blank_line = 0
        all_samples = []
        src_tokens = []
        targets = []
        for line in train_file:
            if line.startswith('-DOCSTART-'):
                continue
            elif line == '\n':
                blank_line += 1
                if blank_line == 2:
                    if len(src_tokens) == 0:
                        blank_line = 1
                        continue
                    blank_line = 1

                    feaMat = utils.get_context_mat(src_tokens, config)
                    label = numpy.zeros((len(targets),), dtype='int32')
                    for tid, target in enumerate(targets):
                        target_parts = target.split('-')
                        if len(target_parts) < 2 or target_parts[1] != self.require_type:
                            label[tid] = config['Tags']['O']
                        else:
                            label[tid] = config['Tags'][target_parts[0]]
                        # elif target_parts[0] == 'B':
                        #     label[tid] = config['Tags']['B']
                        # elif target_parts[0] == 'I':
                        #     label[tid] = config['Tags']['I']

                    question = self.require_type
                    self.sentences.append(zip(src_tokens, label))

                    all_samples.append((feaMat, label, question))

                    src_tokens = []
                    targets = []
            elif blank_line == 1:
                if line.strip():
                    parts = line.strip().split(' ')
                    src_tokens.append(parts[0])
                    targets.append(parts[3])

        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.reset()

    def __iter__(self):
        return self

    def _load_gaz_list(self, file):
        words=set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words

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


class TrainConllBatchGetter:
    def __init__(self, file_name, require_type, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.require_type = require_type
        self.sentences = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        blank_line = 0
        all_samples = []
        neg_samples = []
        src_tokens = []
        targets = []
        for line in train_file:
            if line.startswith('-DOCSTART-'):
                continue
            elif line == '\n':
                blank_line += 1
                if blank_line == 2:
                    if len(src_tokens) == 0:
                        blank_line = 1
                        continue
                    blank_line = 1

                    feaMat = utils.get_context_mat(src_tokens, config)
                    label = numpy.zeros((len(targets),), dtype='int32')
                    has_require_type = False
                    for tid, target in enumerate(targets):
                        target_parts = target.split('-')
                        if len(target_parts) < 2 or target_parts[1] != self.require_type:
                            label[tid] = config['Tags']['O']
                        else:
                            label[tid] = config['Tags'][target_parts[0]]
                            has_require_type = True
                        # elif target_parts[0] == 'B':
                        #     label[tid] = config['Tags']['B']
                        # elif target_parts[0] == 'I':
                        #     label[tid] = config['Tags']['I']

                    question = self.require_type
                    self.sentences.append(zip(src_tokens, label))

                    if has_require_type:
                        all_samples.append((feaMat, label, question))
                    else:
                        neg_samples.append((feaMat, label, question))

                    src_tokens = []
                    targets = []
            elif blank_line == 1:
                if line.strip():
                    parts = line.strip().split(' ')
                    src_tokens.append(parts[0])
                    targets.append(parts[3])

        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)
        self.all_samples.extend(neg_samples[0:self.sample_num])
        self.sample_num = len(self.all_samples)
        self.reset()

    def __iter__(self):
        return self

    def _load_gaz_list(self, file):
        words=set()
        with codecs.open(file, mode='rb', encoding='utf-8') as f:
            for line in f:
                words.add(line.strip())
        return words

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





class ConllBoundaryPerformance(object):
    def __init__(self, vocab, dataset=None):
        self._vocab = vocab
        # self.sentences = SenGetter('data/conll2003/bio_eng.testb', 'ORG', 1, False)
        self.dataset = dataset
        if dataset:
            self.sentences = dataset
        else:
            self.sentences = SenGetter('data/ttt', 'ORG', 1, False)
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

    def extract(self, label_lst):
        B = config['Tags']['B']
        O = config['Tags']['O']
        I = config['Tags']['I']
        if config['bioes']:
            E = config['Tags']['E']
            S = config['Tags']['S']
        else:
            E = None
            S = None
        mentions = []
        label_lst.append(O)
        has_start = False
        has_end = False
        start = 0
        end = 0
        end_is_B = False
        for i, label in enumerate(label_lst):
            if label == B or label == S:
                if not has_start:
                    start = i
                    has_start = True
                elif not has_end:
                    end = i
                    end_is_B = True
                    has_end = True
            if label == O:
                if has_start and not has_end:
                    end = i
                    end_is_B = False
                    has_end = True
            if has_start and has_end:
                mentions.append('_'.join((str(start), str(end))))
                has_start = False
                has_end = False
                if end_is_B:
                    start = end
                    has_start = True
        return set(mentions)



    def evaluate(self, i, label, rec, out_stream=None, pr=True):
        mention_lab = self.extract(label)
        mention_rec = self.extract(rec)
        if pr:
            # print i, mention_rec
            labels = []
            lst = self.sentences.sentences[i]
            for id, token in enumerate(lst):
                out_stream.write('{'+str(id)+' '+token[0]+' '+token[1]+'} ')
                labels.append(token[1])
            out_stream.write('\t')
            types_labels = set([])
            for mention_type in ['LOC', 'PER', 'ORG']:
                one_labels = []
                for tid, target in enumerate(labels):
                    target_parts = target.split('-')
                    if len(target_parts) < 2 or target_parts[1] != mention_type:
                        one_labels.append(config['Tags']['O'])
                    elif target_parts[0] == 'B':
                        one_labels.append(config['Tags']['B'])
                    elif target_parts[0] == 'I':
                        one_labels.append(config['Tags']['I'])
                types_labels.update(self.extract(one_labels))
            for item in mention_rec:
                out_stream.write(item+str(item in types_labels)+' ')
            out_stream.write('\n')
        if self.dataset is not None:
            lst = self.sentences.sentences[i]
            for id, token in enumerate(lst):
                out_stream.write('{' + str(id) + ' ' + token[0] + '} ')
            out_stream.write('\t')
            for item in mention_rec:
                out_stream.write(item+' ')
            out_stream.write('\n')


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
        print('p={},r={}, f={}'.format(p*100, r*100, f))
        return (f, p, r)



def prepro(file_name, out_file):
    train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
    out_file = codecs.open(out_file, mode='wb', encoding='utf-8')

    last = train_file.readline()
    out_file.write(last)
    for line in train_file:
        if line != '\n' and not line.startswith('-DOCSTART-'):
            parts = line.strip().split(' ')
            ner_tag = parts[3].split('-')

            if ner_tag[0] != 'O' and ner_tag[0] != 'B':
                if last != '\n' and not last.startswith('-DOCSTART-'):
                    last_parts = last.strip().split(' ')
                    last_ner_tag = last_parts[3].split('-')
                    if last_ner_tag[0] == 'O':
                        parts[3] = 'B' + parts[3][1:]
                        out = ' '.join(parts)
                        out_file.write(out + '\n')
                    else:
                        if last_ner_tag[1] == ner_tag[1]:
                            out_file.write(line)
                        else:
                            parts[3] = 'B' + parts[3][1:]
                            out = ' '.join(parts)
                            out_file.write(out + '\n')
                else:
                    last_ner_tag = None
                    parts[3] = 'B' + parts[3][1:]
                    out = ' '.join(parts)
                    out_file.write(out + '\n')

            else:
                out_file.write(line)

        else:
            out_file.write(line)
        last = line


    train_file.close()
    out_file.close()


def extract(label_lst):
    B = config['Tags']['B']
    O = config['Tags']['O']
    I = config['Tags']['I']
    if config['bioes']:
        E = config['Tags']['E']
        S = config['Tags']['S']
    else:
        E = None
        S = None

    mentions = []
    label_lst.append(O)
    has_start = False
    has_end = False
    start = 0
    end = 0
    end_is_B = False
    for i, label in enumerate(label_lst):
        if label == B or label == S:
            if not has_start:
                start = i
                has_start = True
            elif not has_end:
                end = i
                end_is_B = True
                has_end = True
        if label == O:
            if has_start and not has_end:
                end = i
                end_is_B = False
                has_end = True
        if has_start and has_end:
            mentions.append('_'.join((str(start), str(end))))
            has_start = False
            has_end = False
            if end_is_B:
                start = end
                has_start = True
    return set(mentions)

class SenGetter(object):
    def __init__(self, file_name, require_type, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._unmatch = 0
        self.require_type = require_type
        self._vocabs = config['Vocabs']
        self._vocab_char = config['CharVoc']
        self._max_char_len = config['max_char']
        self._use_char_conv = config['use_char_conv']
        self._fea_pos = config['fea_pos']
        self._use_gaz = config['use_gaz']
        self.sentences = []
        self.cursor = 0
        train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
        blank_line = 0
        all_samples = []
        src_tokens = []
        targets = []
        for line in train_file:
            if line.startswith('-DOCSTART-'):
                continue
            elif line == '\n':
                blank_line += 1
                if blank_line == 2:
                    if len(src_tokens) == 0:
                        blank_line = 1
                        continue
                    blank_line = 1
                    self.sentences.append(zip(src_tokens, targets))

                    src_tokens = []
                    targets = []
            elif blank_line == 1:
                if line.strip():
                    parts = line.strip().split(' ')
                    src_tokens.append(parts[0])
                    targets.append(parts[3])

        train_file.close()
        self.all_samples = all_samples
        self.sample_num = len(self.all_samples)



if __name__ == '__main__':
    # config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4, 'E': 5, 'S': 6}
    # config['misc'] = False
    # config['bioes'] = True
    # config['use_gaz'] = False
    # onto_notes = OntoNotesGetter('data/OntoNotes/test.json', '/organization/company', 1, False)
    # p = ConllBatchGetter('data/conll2003/bioes_eng.train', 'PER', 1, True)
    # pernam_batch_getter = TrainConllBatchGetter('data/conll2003/bioes_eng.train', 'PER', 1, True)
    # pass
    batch_getter = OntoNotesFGGetter('data/OntoNotes/test.json', utils.get_ontoNotes_train_types(),
                                     fg_config['batch_size'], True)

    print('finish load')
    for this_batch in batch_getter:
        pass
    print('finish read')
