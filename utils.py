# coding: utf-8
import torch
import numpy as np
from torch import nn
from numpy import linalg as LA
import os
import codecs


def switch(vec1, vec2, mask):
    """
    switch function for pytorch

    args:
        vec1 (any size) : input tensor corresponding to 0
        vec2 (same to vec1) : input tensor corresponding to 1
        mask (same to vec1) : input tensor, each element equals to 0/1
    return:
        vec (*)
    """
    catvec = torch.cat([vec1.view(-1, 1), vec2.view(-1, 1)], dim=1)
    switched_vec = torch.gather(catvec, 1, mask.long().view(-1, 1))
    return switched_vec.view(-1)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    # bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    # nn.init.uniform(input_linear.weight, -bias, bias)
    init_weight(input_linear.weight)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_weight(weight):
    bias = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)


def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            param.data.zero_()
            param.data[rnn.hidden_size: 2 * rnn.hidden_size] = 1
            param.data.zero_()
            param.data[rnn.hidden_size: 2 * rnn.hidden_size] = 1
        elif 'weight' in name:
            init_weight(param)


def log_grad_ratio(logger, step, lst):
    for before_step, after_step in lst:
        for before, after in zip(before_step, after_step):
            if before[0] == after[0]:
                tag = before[0]
                value = LA.norm(after[1] - before[1]) / LA.norm(before[1])
                tag = tag.replace('.', '/')
                if value is not None:
                    logger.scalar_summary(tag + '/grad_ratio', value, step)

def get_context_mat(src_tokens, config):
    max_char_len = config['max_char']
    use_gaz = config['use_gaz']
    use_char_conv = config['use_char_conv']
    vocabs = config['Vocabs']
    vocab_char = config['CharVoc']

    if use_gaz:
        # gazdir = config['GazetteerDir']
        # gaz_names = config['Gazetteers']
        # gazetteers = []
        # for (id, gaz) in enumerate(gaz_names):
        #     gazfile = os.path.join(gazdir, gaz)
        #     gazetteers.append(load_gaz_list(gazfile))
        gazetteers = config['load_gaz']

    fea_len = 1
    if use_gaz:
        fea_len += len(gazetteers)
    if use_char_conv:
        fea_len += max_char_len * 2

    feaMat = np.zeros((len(src_tokens), fea_len), dtype='int32')
    for (lid, token) in enumerate(src_tokens):
        for (i, voc) in enumerate(vocabs):
            wid = voc.getID(token.lower())
            feaMat[lid, i] = wid
        curr_end = len(vocabs)
        if use_gaz:
            gazStart = len(vocabs)
            for (id, gaz) in enumerate(gazetteers):
                if token.lower() in gaz:
                    feaMat[lid, id + gazStart] = 1
            curr_end += len(gazetteers)
        if use_char_conv:
            word = token
            chStart = curr_end
            chMaskStart = chStart + max_char_len
            for i in range(len(word)):
                if i >= max_char_len:
                    break
                feaMat[lid, chStart + i] = vocab_char.getID(word[i])
                feaMat[lid, chMaskStart + i] = 1

    return feaMat


def get_fg_mat(src_tokens, config):
    vocabs = config['Vocabs']

    fea_len = 1

    feaMat = np.zeros((len(src_tokens), fea_len), dtype='int32')
    for (lid, token) in enumerate(src_tokens):
        for (i, voc) in enumerate(vocabs):
            wid = voc.getID(token.lower())
            feaMat[lid, i] = wid

    return feaMat

def get_ctx_mat(src_tokens, config):
    vocabs = config['Vocabs']

    fea_len = 1


    feaMat = np.zeros((config['ctx_window_size'], fea_len), dtype='int32')
    for (lid, token) in enumerate(src_tokens):
        for (i, voc) in enumerate(vocabs):
            wid = voc.getID(token.lower())
            feaMat[lid, i] = wid

    return feaMat



def load_gaz_list(file):
    words=set()
    with codecs.open(file, mode='rb', encoding='utf-8') as f:
        for line in f:
            words.add(line.strip())
    return words


def get_ontoNotes_type_lst():
    test_lst = [
     '/other/body_part', '/location/geography', '/other/food',
     '/person/business',
     '/organization/military', '/other/health',
     '/location/park', '/organization/company', '/other/scientific',
     '/other/product', '/person/education', '/person/doctor',
     '/person/athlete', '/location/structure',
     '/person/political_figure', '/other/heritage',
     '/organization/government', '/other/living_thing', '/person/military',
     '/location/celestial', '/other/sports_and_leisure',
     '/other/art', '/other/currency', '/person/title', '/other/religion',
     '/person/legal', '/person/artist',
     '/location/transit', '/organization/political_party',
     '/location/city', '/organization/education',
     '/organization/sports_team', '/other/event', '/other/legal', '/location/country']

    return test_lst



def get_ontoNotes_train_types():
    lst = ['/location', '/other/body_part', '/organization/company', '/other/event/accident', '/other/event/holiday',
           '/person/military',
           '/location/structure', '/organization/company/news', '/location/structure/theater',
           '/person/artist/director',
           '/other/product/weapon', '/other/award', '/location/transit/road', '/organization/sports_team',
           '/other/product/software',
           '/location/country', '/other/product/computer', '/other/food', '/location/geography',
           '/other/language/programming_language',
           '/other/art/broadcast', '/other/living_thing/animal', '/location/structure/hotel', '/other/scientific',
           '/person/artist/music',
           '/other/product', '/other/product/mobile_phone', '/location/structure/government',
           '/location/geography/island',
           '/organization/sports_league', '/other/currency', '/other/language', '/person/artist/author',
           '/organization/music',
           '/location/transit/railway', '/location/geography/body_of_water', '/organization/stock_exchange',
           '/other/health/treatment', '/other/internet', '/location/park', '/location/structure/hospital',
           '/other/event/natural_disaster', '/person/doctor', '/other/art/stage', '/person/artist/actor',
           '/person/athlete', '/location/transit', '/other/art/writing', '/other/living_thing',
           '/person/political_figure', '/organization/government', '/other/event/sports_event', '/location/celestial',
           '/other/sports_and_leisure', '/location/transit/bridge', '/organization/company/broadcast',
           '/location/structure/restaurant', '/person/artist', '/other', '/other/product/car', '/location/city',
           '/other/event', '/other/legal', '/other/religion', '/other/event/protest', '/location/structure/airport',
           '/other/event/violent_conflict', '/other/art/music', '/organization/transit', '/organization/military',
           '/person', '/other/health', '/organization', '/other/event/election', '/person/religious_leader',
           '/other/heritage', '/other/health/malady', '/other/art', '/other/supernatural', '/location/geograpy/island',
           '/person/title', '/person/legal', '/location/geograpy', '/location/geography/mountain',
           '/organization/political_party',
           '/location/structure/sports_facility', '/organization/education', '/person/coach', '/other/art/film']
    return lst





