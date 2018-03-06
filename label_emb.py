# coding: utf-8

import codecs


def data_prepro(raw_data, out_data):
    raw_file = codecs.open(raw_data, mode='rb', encoding='utf-8')
    out_file = codecs.open(out_data, mode='wb', encoding='utf-8')
    for line in raw_file:
        line = line.strip()
        if line:
            parts = line.split('|||')
            src_tokens = parts[0].split(' ')
            src_tokens = [token.split('#')[0] for token in src_tokens]
            tgt_tokens = parts[1].split(' ')
        mention_rep = []
        stack = []
        word_no = 0
        for tgt in tgt_tokens:
            if tgt == 'X':
                if len(stack) == 0:
                    mention_rep.append(src_tokens[word_no])
                word_no += 1
            elif tgt.startswith('('):
                stack.append(tgt[1:-4])
            elif tgt.startswith(')'):
                type = stack.pop()
                if len(stack) == 0:
                    mention_rep.append('%'+type+'%')
        out_file.write(' '.join(mention_rep))
        out_file.write('\n')
        print(' '.join(mention_rep))
    out_file.close()


if __name__ == '__main__':
    data_prepro('data/train', 'data/train_label')





