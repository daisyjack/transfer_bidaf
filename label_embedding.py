import torch
from bidaf import LoadEmbedding
from configurations import config
from tensorboardX import SummaryWriter



def gen_embedding(voc, embedding_tensor, summary_writer, write_file=True):
    if write_file:
        out_f = open('data/label_embedding/glove_6B_embedding.txt', mode='w')

    emb_lst = []
    meta = []

    with open('data/label_embedding/prototypes_Onto.csv') as lb:
        for line in lb:
            line = line.strip()
            parts = line.split('\t')
            label_name = None
            label_words = []
            for i, part in enumerate(parts):
                if i == 0:
                    label_name = part
                else:
                    label_words.extend(part.split('_'))
            embedding = torch.zeros(300)
            for word in label_words:
                word_id = voc.getID(word.lower())
                embedding += embedding_tensor[word_id]
            embedding = embedding / len(label_words)
            emb_lst.append(embedding.unsqueeze(0))
            meta.append(label_name)
            if write_file:
                out_f.write('{} '.format(label_name))
                for num_pos, a in enumerate(embedding):
                    if num_pos < 299:
                        out_f.write('%.5f ' % a)
                    else:
                        out_f.write('%.5f' % a)
                out_f.write('\n')
        # summary_writer.add_embedding(torch.cat(emb_lst, 0), metadata=meta)
        if write_file:
            out_f.close()

if __name__ == '__main__':
    emb = LoadEmbedding('res/glove.6B.300d.txt')
    voc = config['onto_word_id']
    # summary_writer = SummaryWriter('ner_logs17')
    gen_embedding(voc, emb.get_embedding_tensor(), None, True)


