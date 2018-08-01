# coding: utf-8
from __future__ import print_function
import codecs

from logger import Logger

def to_np(x):
    return x.data.cpu().numpy()

class Vocab(object):
    def __init__(self, vocfile, unk_id, pad_id):
        self._word2id = {}
        self._id2word = {}
        self.unk_id = unk_id
        self.padding_id = pad_id
        self._voc_name = vocfile
        with codecs.open(vocfile, mode='rb', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) != 2:
                        print('illegal voc line %s' % line)
                        continue
                    id = int(parts[1])
                    # if parts[0] in self._word2id:
                    #     print(parts[0])
                    self._word2id[parts[0]] = id
                    self._id2word[id] = parts[0]
        self._vocab_size = max(self._word2id.values()) + 1
        self.unk = self._id2word[self.unk_id]
        self.PADDING = self._id2word[self.padding_id]
        if self._vocab_size != len(self._word2id):
            print('in vocab file {}, vocab_max {} not equal to vocab count {}, maybe empty id or others' \
                .format(vocfile, self._vocab_size, len(self._word2id)))

    def __str__(self):
        return self._voc_name

    def getVocSize(self):
        return self._vocab_size

    def getWord(self, id):
        return self._id2word[id] if id in self._id2word else self.unk

    def getID(self, word):
        return self._word2id[word] if word in self._word2id else self.unk_id

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

class Config(object):
    def __init__(self):
        self.dic = {}

    def __setitem__(self, key, value):
        # if key == 'EmbSizes':
        #     print '\n<<<<<', 'EmbSizes'
        self.dic[key] = value

    def __getitem__(self, item):
        # if item == 'EmbFiles':
        #     print '\n<<<', 'EmbFiles'
        # if item == 'EmbSizes':
        #     print '\n<<<<<', 'EmbSizes'
        return self.dic[item]


def get_conf_base():
    config = {}
    config['early_stop'] = 5
    config['USE_CUDA'] = True
    config['weight_scale'] = 0.05
    config['cuda_num'] = 2
    config['max_seq_length'] = 4
    config['max_label_length'] = 100
    config['batch_size'] = 64  #32
    config['hidden_size'] = 100
    config['decoder_layers'] = 1
    config['encoder_filter_num'] = 400
    config['encoder_outputs_size'] = config['hidden_size']
    config['decoder_output_size'] = 18
    config['clip_norm'] = 5.0
    config['beam_size'] = 10
    config['EOS_token'] = 2
    config['PAD_token'] = 0
    config['UNK_token'] = 1
    config['X'] = 17
    config['att_mode'] = 'general'
    config['OutTags'] = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4, 'E': 5, 'S': 6}
    config['WordId'] = Vocab('res/voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['filter_size'] = 5  # 3
    config['highway_num_layers'] = 2
    config['save_freq'] = 30
    config['dropout'] = 0.2  # 0.25
    config['use_dropout'] = True
    # config['dropout'] = 0.9
    config['multi_cuda'] = [0, 1]
    config['use_multi'] = True

    config['max_char'] = 25
    config['char_emb_dim'] = 8  # 50
    config['out_channel_dims'] = 100
    config['CharVoc'] = Vocab('res/char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    config['WordPos'] = 1
    config['EmbNames'] = ('Words', 'CAPS', 'POS', 'NER')
    # config['EmbNames'] = ('Words',)
    config['EmbSizes'] = (100, 30, 30, 30)
    config['const-emb'] = (False, False, False, False)
    capsVoc = Vocab('res/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    posVoc = Vocab('res/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    nerVoc = Vocab('res/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Vocabs'] = (config['WordId'],)  # (config['WordId'], capsVoc, posVoc, nerVoc)#(config['WordId'],)  # (config['WordId'], capsVoc, posVoc, nerVoc)

    config['EmbFiles'] = ('res/embedding.txt', None, None, None)

    config['use_char_conv'] = True
    config['use_gaz'] = False
    config['Gazetteers'] = ('PER', 'LOC', 'ORG', 'VEH', 'FAC', 'WEA')
    config['GazetteerDir'] = 'res/gazetteers'
    config['gaz_emb_dim'] = 30
    config['max_epoch'] = 100
    config['transition'] = False
    config['question_alone'] = False
    config['bioes'] = True
    config['onto_word_id'] = Vocab('res/glove_6B_voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])

    return config


def get_fg_config():
    config = {}
    config['early_stop'] = 10
    config['USE_CUDA'] = True
    config['weight_scale'] = 0.05
    config['cuda_num'] = 2
    config['max_seq_length'] = 4
    config['max_label_length'] = 100
    config['batch_size'] = 64  # 32
    config['hidden_size'] = 100
    config['decoder_layers'] = 1
    config['encoder_filter_num'] = 400
    config['encoder_outputs_size'] = config['hidden_size']
    config['decoder_output_size'] = 18
    config['clip_norm'] = 1.0
    config['beam_size'] = 10
    config['EOS_token'] = 2
    config['PAD_token'] = 0
    config['UNK_token'] = 1
    config['X'] = 17
    config['att_mode'] = 'general'
    config['OutTags'] = Vocab('res/ner_xx', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Tags'] = {'<PADDING>': 0, '<START>': 1, 'B': 2, 'I': 3, 'O': 4, 'E': 5, 'S': 6}
    config['WordId'] = Vocab('res/voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['onto_word_id'] = Vocab('res/glove_840B_voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['type_id'] = Vocab('res/onto/zero_type_voc.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['filter_size'] = 5  # 3
    config['highway_num_layers'] = 2
    config['save_freq'] = 50
    config['dropout'] = 0.2  # 0.25
    config['use_dropout'] = True
    # config['dropout'] = 0.9
    config['multi_cuda'] = [0, 1]
    config['use_multi'] = True

    config['max_char'] = 25
    config['char_emb_dim'] = 8  # 50
    config['out_channel_dims'] = 100
    config['CharVoc'] = Vocab('res/char.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['fea_pos'] = (0, 2, 3, 4)  # 0,2,3,5 for coref
    config['WordPos'] = 1
    config['EmbNames'] = ('Words', 'CAPS', 'POS', 'NER')
    # config['EmbNames'] = ('Words',)
    config['EmbSizes'] = (100, 30, 30, 30)
    config['const-emb'] = (False, False, False, False)
    capsVoc = Vocab('res/caps.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    posVoc = Vocab('res/pos.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    nerVoc = Vocab('res/ner.txt', unk_id=config['UNK_token'], pad_id=config['PAD_token'])
    config['Vocabs'] = (config['onto_word_id'],)  # (config['WordId'], capsVoc, posVoc, nerVoc)#(config['WordId'],)  # (config['WordId'], capsVoc, posVoc, nerVoc)

    config['EmbFiles'] = ('res/embedding.txt', None, None, None)

    config['use_char_conv'] = True
    config['use_gaz'] = False
    config['Gazetteers'] = ('PER', 'LOC', 'ORG', 'VEH', 'FAC', 'WEA')
    config['GazetteerDir'] = 'res/gazetteers'
    config['gaz_emb_dim'] = 30
    config['max_epoch'] = 100
    config['transition'] = False
    config['question_alone'] = False
    config['bioes'] = True

    config['ctx_window_size'] = 10
    config['lstm_layers'] = 1
    config['Da'] = 100
    config['topk'] = 3

    return config



def get_conf_ner():
    config = get_conf_base()
    return config


def get_conf_coref():
    config = get_conf_base()
    return config


import os


def get_conf(task, datadir=None, saveto='saved'):
    config = None
    if task == 'ner':
        config = get_conf_ner()
        # config['saveto'] = saveto
        # config['train_data'] = os.path.join(datadir, 'train')
        # config['test_txt'] = os.path.join(datadir, 'dev')
        return config
    elif task == 'coref':
        config = get_conf_coref()
        config['saveto'] = saveto
        config['train_data'] = os.path.join(datadir, 'train.txt')
        config['test_txt'] = os.path.join(datadir, 'dev.txt')
        return config
    else:
        print('unknown task {}'.format(task))
        return None


USE_CUDA = True
max_seq_length = 4
max_label_length = 8
batch_size = 2
hidden_size = 2
n_layers = 2
encoder_outputs_dim = 2
output_size = 3
cuda_num = 0
clip_norm = 1
beam_size = 6
EOS_token = 2
PAD_token = 100
X = 1
att_mode = 'general'

config = get_conf('ner')
fg_config= get_fg_config()
# logger = Logger('./logs')
