# coding: utf-8

import json
import copy
import codecs


def get_type(file_name):
    mention_type = set([])
    with open(file_name) as train_file:
        for line in train_file:
            line = json.loads(line)
            src_tokens = line['tokens']
            mentions = line['mentions']
            for mention in mentions:
                # if len(mention['labels']) == 1:
                mention_type.update(mention['labels'])
    return mention_type


def get_leaf(file_name, out_f):
    train_file = open(file_name, mode='r')
    out_file = open(out_f, mode='w')
    all_samples = []

    for line_no, line in enumerate(train_file):
        line = json.loads(line)
        mentions = line['mentions']
        new_mentions = copy.deepcopy(mentions)

        for i, mention in enumerate(mentions):
            new_mentions[i]['labels'] = []

            for j, label in enumerate(mention['labels']):
                is_parent = False
                for k, other_label in enumerate(mention['labels']):
                    if k != j and (label in other_label):
                        is_parent = True
                if not is_parent:
                    new_mentions[i]['labels'].append(label)
        line['mentions'] = new_mentions
        out_file.write(json.dumps(line)+'\n')

    train_file.close()
    out_file.close()

def get_ontonotes_labels_num(file_name):
    train_file = codecs.open(file_name, mode='rb', encoding='utf-8')
    labels_num = 0

    for line_no, line in enumerate(train_file):
        line = json.loads(line)
        src_tokens = line['tokens']
        mentions = line['mentions']


        for mention in mentions:
            labels_num += len(mention['labels'])
    print(labels_num)


    train_file.close()

def get_voc_size():
    # emb = codecs.open('res/glove.840B.300d.txt', mode='rb', encoding='utf-8')
    # voc = codecs.open('res/glove_voc.txt', mode='wb', encoding='utf-8')
    size = 0
    with codecs.open('res/glove.6B.300d.txt', mode='rb', encoding='utf-8') as emb:
        with codecs.open('res/glove_6B_voc.txt', mode='wb', encoding='utf-8') as voc:

            for line in emb:
                line = line.strip()
                if line:
                    num = size + 92
                    word = line.split(' ')[0]
                    voc.write(word + '\t' + str(num) +'\n')
                    size += 1
    # voc.close()



    # emb.close()
    print(size)



if __name__ == '__main__':
    # get_ontonotes_labels_num('data/OntoNotes/test.json')
    # test_mention_type = get_type('data/OntoNotes/test.json')
    # train_mention_type = get_type('data/OntoNotes/train.json')
    # print(train_mention_type, len(train_mention_type))
    # print(test_mention_type, len(test_mention_type))
    # print(train_mention_type.difference(test_mention_type), len(train_mention_type.difference(test_mention_type)))
    # print(test_mention_type.difference(train_mention_type), len(test_mention_type.difference(train_mention_type)))
    # print(train_mention_type.intersection(test_mention_type), len(train_mention_type.intersection(test_mention_type)))
    # get_leaf('data/OntoNotes/train.json', 'data/OntoNotes/leaf_train.json')
    get_voc_size()
