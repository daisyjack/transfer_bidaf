from torch import nn
import utils
import torch
from configurations import fg_config



def get_type_lst(data):
    type_lst = None
    if data == 'onto':
        type_lst = utils.get_ontoNotes_train_types()
    elif data == 'wiki':
        type_lst = utils.get_wiki_types()
    elif data == 'bbn':
        type_lst = utils.get_bbn_types()
    return type_lst

class NZSigmoidLoss(nn.Module):
    def __init__(self, hidden_size, word_emb_size):
        super(NZSigmoidLoss, self).__init__()
        require_type_lst = get_type_lst(fg_config['data'])
        self.weight = nn.Parameter(torch.zeros(len(require_type_lst), hidden_size*2+word_emb_size))
        utils.init_weight(self.weight)

    def forward(self, ctx_rep, men_rep, labels):
        """
        args:
        :param ctx_rep: (B, hidden_size*2)
        :param men_rep: (B, word_emb)
        :param labels: （B, 89）
        :return:
        """
        if fg_config['att'] == 'label_att':
            # ctx_rep: (89, B, hidden_size * 2)
            types_len = self.weight.size(0)
            batch_size = men_rep.size(0)
            word_emb = men_rep.size(1)
            men_rep = men_rep.unsqueeze(0).expand(types_len, batch_size, word_emb)  # (89, B, word_emb)
            ctx_men = torch.cat((ctx_rep, men_rep), 2)  # (89, B, hidden_size * 2+word_emb)
            weight = self.weight.unsqueeze(2)  # (89, hidden_size * 2+word_emb, 1)
            logits = torch.bmm(ctx_men, weight).squeeze(2).transpose(0, 1)  # (B, 89)

        else:
            ctx_rep = ctx_rep[0]
            logits = torch.mm(torch.cat((ctx_rep, men_rep), 1), self.weight.transpose(0, 1))



        # logits = torch.clamp(logits, max=16)
        logits = torch.sigmoid(logits)  # (B, 89)
        loss = torch.sum(-labels*torch.log(logits+1e-6)-(1-labels)*torch.log(1-logits+1e-6)) / labels.size(0)
        return loss, logits