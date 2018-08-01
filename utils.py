# coding: utf-8
import torch
import numpy as np
from torch import nn
from numpy import linalg as LA
import os
import codecs
import pickle


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


def get_fg_mat(src_tokens, config, is_type=False):
    vocabs = config['Vocabs']
    type_vocab = config['type_id']

    fea_len = 1

    feaMat = np.zeros((len(src_tokens), fea_len), dtype='int32')
    if is_type:
        for (lid, token) in enumerate(src_tokens):
            wid = type_vocab.getID(token)
            feaMat[lid, 0] = wid
    else:
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

def load_embedding(emb_file):
    with open(emb_file, 'rb') as emb_file:
        emb = pickle.load(emb_file)
    return emb


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

def check_depth(label, req_depth):
    depth = len(label.split('/')) - 1
    if depth == req_depth:
        return True
    else:
        return False

def get_ontoNotes_train_types(req_depth=None):
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
    req_lst = []
    for type in lst:
        if check_depth(type, req_depth):
            req_lst.append(type)
    if req_depth is None:
        req_lst = lst

    return req_lst

def get_wiki_types(req_depth=None):
    lst = ['/product/mobile_phone', '/location/country', '/event/military_conflict', '/transit', '/broadcast_program',
           '/person/monarch', '/play', '/game', '/broadcast_network', '/military', '/visual_art/color', '/law',
           '/location/county', '/person/author', '/product/airplane', '/event/protest', '/metropolitan_transit/transit_line',
           '/person/architect', '/livingthing/animal', '/newspaper', '/transportation/road', '/building/airport',
           '/person/musician', '/product/spacecraft', '/location/city', '/event', '/government_agency', '/building/library',
           '/product/ship', '/person/engineer', '/person/politician', '/art', '/building/sports_facility',
           '/person/terrorist', '/body_part', '/organization/company', '/education/educational_degree',
           '/person/religious_leader', '/people/ethnicity', '/person/soldier', '/train', '/art/film',
           '/medicine/symptom', '/building/hotel', '/internet/website', '/organization/fraternity_sorority',
           '/finance/currency', '/person/director', '/geography/glacier', '/product/computer',
           '/organization/terrorist_organization', '/time', '/building/power_station', '/government/political_party',
           '/computer/algorithm', '/location/cemetery', '/person/doctor', '/disease', '/event/attack', '/event/election',
           '/location/bridge', '/education/department', '/location/body_of_water', '/organization/sports_team', '/god',
           '/building/dam', '/broadcast/tv_channel', '/product/instrument', '/product', '/chemistry', '/organization',
           '/person/actor', '/biology', '/finance/stock_exchange', '/title', '/news_agency', '/product/camera',
           '/person/coach', '/event/natural_disaster', '/medicine/medical_treatment', '/event/sports_event',
           '/building/theater', '/organization/sports_league', '/park', '/person/athlete', '/organization/airline',
           '/food', '/event/terrorist_attack', '/rail/railway', '/living_thing', '/computer/programming_language',
           '/product/car', '/award', '/building/hospital', '/organization/educational_institution',
           '/government/government', '/software', '/music', '/building/restaurant', '/person/artist',
           '/product/engine_device', '/medicine/drug', '/person', '/product/weapon', '/geography/island', '/building',
           '/language', '/written_work', '/religion/religion', '/location/province', '/geography/mountain', '/location',
           '/astral_body']
    req_lst = []
    for type in lst:
        if check_depth(type, req_depth):
            req_lst.append(type)
    if req_depth is None:
        req_lst = lst

    return req_lst

def get_bbn_types(req_depth=None):
    lst = ['/ORGANIZATION/POLITICAL', '/SUBSTANCE', '/LOCATION/CONTINENT', '/FACILITY', '/ORGANIZATION/EDUCATIONAL',
           '/ORGANIZATION/GOVERNMENT', '/EVENT', '/FACILITY/ATTRACTION', '/WORK_OF_ART/BOOK', '/GPE/COUNTRY',
           '/SUBSTANCE/CHEMICAL', '/GPE/CITY', '/ORGANIZATION/HOTEL', '/PLANT', '/GPE', '/EVENT/WAR',
           '/LOCATION/LAKE_SEA_OCEAN', '/LOCATION', '/GPE/STATE_PROVINCE', '/FACILITY/BUILDING', '/PRODUCT/WEAPON',
           '/ORGANIZATION/HOSPITAL', '/PRODUCT', '/FACILITY/HIGHWAY_STREET', '/DISEASE', '/EVENT/HURRICANE', '/LAW',
           '/LOCATION/RIVER', '/CONTACT_INFO', '/PERSON', '/ANIMAL', '/PRODUCT/VEHICLE', '/WORK_OF_ART',
           '/CONTACT_INFO/url', '/ORGANIZATION/CORPORATION', '/LOCATION/REGION', '/WORK_OF_ART/SONG', '/SUBSTANCE/FOOD',
           '/GAME', '/LANGUAGE', '/SUBSTANCE/DRUG', '/FACILITY/BRIDGE', '/FACILITY/AIRPORT', '/ORGANIZATION',
           '/ORGANIZATION/RELIGIOUS', '/ORGANIZATION/MUSEUM', '/WORK_OF_ART/PLAY']
    req_lst = []
    for type in lst:
        if check_depth(type, req_depth):
            req_lst.append(type)
    if req_depth is None:
        req_lst = lst

    return req_lst


def wiki_short2full_patch():
    lst = ['/medicine', '/religion', '/visual_art', '/rail', '/geography', '/computer', '/broadcast', '/people',
           '/finance', '/government', '/livingthing', '/education', '/transportation', '/internet',
           '/metropolitan_transit']
    dct = {}
    for type in lst:
        dct[type.split('/')[1]] = type
    return dct




if __name__ == '__main__':
    # lst = get_wiki_types(2)
    # type2 = set([])
    # type1 = get_wiki_types(1)
    # for type in lst:
    #     type2.add('/'+type.split('/')[1])
    # print(len(type2), type2)
    # print(len(type1), type1)
    # dif = type2.difference(set(type1))
    # print(len(dif), dif)
    print(len(get_bbn_types()))






