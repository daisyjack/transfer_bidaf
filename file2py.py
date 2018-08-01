import codecs


def nz_ontoNotes_to_pickle():
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
            l_ctx_start = start - fg_config['ctx_window_size'] if start - fg_config['ctx_window_size'] >= 0 else 0
            l_ctx_end = start
            r_ctx_start = end + 1
            r_ctx_end = r_ctx_start + fg_config['ctx_window_size']
            l_ctx_tokens = src_tokens[l_ctx_start:l_ctx_end]
            r_ctx_tokens = src_tokens[r_ctx_start:r_ctx_end]
            men_mat = utils.get_fg_mat(men_tokens, fg_config)
            l_ctx_mat = utils.get_ctx_mat(l_ctx_tokens, fg_config)
            r_ctx_mat = utils.get_ctx_mat(r_ctx_tokens, fg_config)
            label = numpy.zeros((1, len(require_type_lst)), dtype='int32')
            for i, require_type in enumerate(require_type_lst):
                if require_type in mention['labels']:
                    label[0, i] = 1
            type_mat = utils.get_fg_mat(require_type_lst, fg_config)
            self.all_samples.append(
                [l_ctx_mat, men_mat, r_ctx_mat, len(l_ctx_tokens), len(r_ctx_tokens), label, mention['labels'],
                 type_mat])

    train_file.close()