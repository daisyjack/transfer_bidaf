# coding: utf-8

import torch
import json
import codecs
from configurations import config
import numpy
import string
import random
from batch_getter import pad_tgt_seq, pad_src_seq

def remove_puctuation(word):
    table = str.maketrans({key: None for key in string.punctuation})
    return word.translate(table)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def text2mat(text):
    max_char_len = config['max_char']
    wordId = config['Vocabs'][0]
    vocab_char = config['CharVoc']

    fea_len = 1 + max_char_len * 2
    src_tokens = text.split()
    feaMat = numpy.zeros((len(src_tokens), fea_len), dtype='int32')
    for (lid, token) in enumerate(src_tokens):
        word_no_punc = remove_punc(token)
        token_lower = word_no_punc.lower()
        feaMat[lid, 0] = wordId.getID(token_lower)
        chStart = 1
        chMaskStart = chStart + max_char_len
        for i in range(len(word_no_punc)):
            if i >= max_char_len:
                break
            feaMat[lid, chStart + i] = vocab_char.getID(word_no_punc[i])
            feaMat[lid, chMaskStart + i] = 1
    return feaMat




class SquadLoader:
    def __init__(self, file_path, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.wordId = config['Vocabs'][0]

        self.all_samples = []
        self.cursor = 0
        with open(file_path, mode='r') as f:
            dataset = json.load(f)
            dataset = dataset['data']
        self.dataset = dataset
        self.questions = []
        self.paragraphs = []
        for art_id, article in enumerate(dataset):
            article_tmp = []
            for para_id, paragraph in enumerate(article['paragraphs']):
                context = paragraph['context']
                article_tmp.append(text2mat(context))
                for question in paragraph['qas']:
                    if question['answers'][0]['answer_start'] == 0:
                        start_word_num = 0
                    elif context[question['answers'][0]['answer_start']-1] == ' ':
                        start_word_num = len(context[:question['answers'][0]['answer_start']].split())
                    else:
                        start_word_num = len(context[:question['answers'][0]['answer_start']].split()) - 1
                    end_word_num = start_word_num + len(question['answers'][0]['text'].split()) - 1
                    self.questions.append({'question': text2mat(question['question']), 'id': question['id'], 'ans_start': start_word_num,
                                           'ans_end': end_word_num, 'art_id': art_id, 'para_id': para_id})
            self.paragraphs.append(article_tmp)

        self.sample_num = len(self.questions)

        self.reset()

    def __iter__(self):
        return self

    # 一个epoch后reset
    def reset(self):
        if self.shuffle:
            random.shuffle(self.questions)
        self.cursor = 0


    def __next__(self):
        if self.cursor < self.sample_num:
            required_batch = self.questions[self.cursor:self.cursor+self.batch_size]
            self.cursor += self.batch_size
            final_batch = {'ids': [], 'art_ids': [], 'para_ids': []}
            contexts = []
            questions = []
            questions_lens = []
            contexts_lens = []
            ans_starts = []
            ans_ends = []
            for question in required_batch:
                final_batch['ids'].append(question['id'])
                final_batch['art_ids'].append(question['art_id'])
                final_batch['para_ids'].append(question['para_id'])
                questions.append(question['question'])
                context = self.paragraphs[question['art_id']][question['para_id']]
                contexts.append(context)
                questions_lens.append(question['question'].shape[0])
                contexts_lens.append(context.shape[0])
                ans_starts.append(question['ans_start'])
                ans_ends.append(question['ans_end'])
            contexts_padded = [pad_src_seq(context, max(contexts_lens))[numpy.newaxis, ...] for context in contexts]
            contexts_tensor = torch.from_numpy(numpy.concatenate(contexts_padded, axis=0)).type(torch.LongTensor)
            questions_padded = [pad_src_seq(s, max(questions_lens))[numpy.newaxis, ...] for s in questions]
            questions_tensor = torch.from_numpy(numpy.concatenate(questions_padded, axis=0)).type(torch.LongTensor)
            final_batch['contexts'] = contexts_tensor
            final_batch['questions'] = questions_tensor
            final_batch['con_lens'] = contexts_lens
            final_batch['q_lens'] = questions_lens
            start_mat = numpy.zeros((len(required_batch), max(contexts_lens)), dtype='int32')
            for i, num in enumerate(ans_starts):
                start_mat[i, num] = 1
            end_mat = numpy.zeros((len(required_batch), max(contexts_lens)), dtype='int32')
            for i, num in enumerate(ans_ends):
                end_mat[i, num] = 1
            final_batch['start'] = torch.from_numpy(start_mat).type(torch.FloatTensor)
            final_batch['end'] = torch.from_numpy(end_mat).type(torch.FloatTensor)
            final_batch['ans_start'] = ans_starts
            final_batch['ans_end'] = ans_ends
            return final_batch
        else:
            raise StopIteration("out of list")







if __name__ == '__main__':
    a = SquadLoader('data/SQuAD/train-v1.1.json', 8, True)
    for b in a:
        pass
    print(a.questions[439])



