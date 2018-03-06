# coding: utf-8
import codecs

def bio2bioes(f_name, out_name):
    in_file = codecs.open(f_name, mode='rb', encoding='utf-8')
    out_file = codecs.open(out_name, mode='wb', encoding='utf-8')
    this_label = 'O'
    this_line = None
    for next_line in in_file:
        if this_line is None:
            this_line = next_line
            continue
        elif this_line.startswith('-DOCSTART-'):
            out_file.write(this_line)
            this_line = next_line
            continue
        elif this_line == '\n':
            out_file.write(this_line)
            this_line = next_line
            continue
        else:
            this_parts = this_line.strip().split(' ')
            this_label_parts = this_parts[3].split('-')
            if len(this_label_parts) < 2:
                this_label = 'O'
            else:
                this_label = this_label_parts[0]
            if next_line.startswith('-DOCSTART-') or next_line == '\n':
                next_label = 'O'
            else:
                next_parts = next_line.strip().split(' ')
                next_label_parts = next_parts[3].split('-')
                if len(next_label_parts) < 2:
                    next_label = 'O'
                else:
                    next_label = next_label_parts[0]

            if this_label == 'I' and next_label != 'I':
                # change I to E
                out_file.write(' '.join(this_parts[0:-1]) + ' ' + '-'.join(['E', this_label_parts[-1]]) + '\n')
            elif this_label == 'B' and next_label != 'I':
                # change B to S
                out_file.write(' '.join(this_parts[0:-1]) + ' ' + '-'.join(['S', this_label_parts[-1]]) + '\n')
            else:
                out_file.write(this_line)
            this_line = next_line
    out_file.close()
    in_file.close()


if __name__ == '__main__':
    f_name = 'data/conll2003/bio_eng.testb'
    out_name = 'data/conll2003/bioes_eng.testb'
    bio2bioes(f_name, out_name)


