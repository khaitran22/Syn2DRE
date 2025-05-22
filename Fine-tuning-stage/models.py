import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import *
import torch.nn.functional as F
from axial_attention import AxialAttention, AxialImageTransformer
import numpy as np
import math
from itertools import accumulate
from torch.nn.parameter import Parameter

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


class AxialTransformer_by_entity(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x


class AxialEntityTransformer(nn.Module):
    def  __init__(self, emb_size = 768, dropout = 0.1, num_layers = 2, dim_index = -1, heads = 8, num_dimensions=2, ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_index = dim_index
        self.heads = heads
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_dimensions = num_dimensions
        self.axial_img_transformer = AxialImageTransformer()
        self.axial_attns = nn.ModuleList([AxialAttention(dim = self.emb_size, dim_index = dim_index, heads = heads, num_dimensions = num_dimensions, ) for i in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size) for i in range(num_layers)] )
        self.lns = nn.ModuleList([nn.LayerNorm(self.emb_size) for i in range(num_layers)])
        self.attn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(num_layers)] )
    
    def forward(self, x):
        for idx in range(self.num_layers):
          x = x + self.attn_dropouts[idx](self.axial_attns[idx](x))
          x = self.ffns[idx](x)
          x = self.ffn_dropouts[idx](x)
          x = self.lns[idx](x)
        return x


class DocREModel(nn.Module):
    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1,):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.mse_fnt = nn.MSELoss()

        self.sizeA = 200

        self.head_extractor = nn.Linear(2 * (config.hidden_size+self.sizeA), emb_size)
        self.tail_extractor = nn.Linear(2 * (config.hidden_size+self.sizeA), emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.gc1 = GraphConvolution(config.hidden_size, self.sizeA)
        self.dropout = nn.Dropout(args.dropout)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                Adj=None,
                teacher_logits=None,
                update_meta=False,
                negative_mask=None
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output=self.dropout(sequence_output)

        # GCN
        Adj = F.normalize(Adj)
        sequence_output_A=torch.relu(self.gc1(sequence_output,Adj))
        sequence_output=torch.cat([sequence_output,sequence_output_A],dim=2)
        
        hts_tensor = torch.cat([torch.tensor(h) for h in hts])
        self_mask = (hts_tensor[:, 0] != hts_tensor[:, 1]).float().view(-1, 1).to(sequence_output)

        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        logits = logits * self_mask

        output = (logits, self.loss_fnt.get_label(logits, num_labels=self.num_labels))

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        else:
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
        
        return output


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # in_features × out_features
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class KD_DocRE(nn.Module):
    def __init__(self, args, config, model, emb_size=1024, block_size=64, num_labels=-1, teacher_model=None):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.loss_fnt = AFLoss(gamma_pos = args.gamma_pos, gamma_neg = args.gamma_neg,)
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.entity_criterion = nn.CrossEntropyLoss()
        self.bin_criterion = nn.CrossEntropyLoss()
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.projection = nn.Linear(emb_size * block_size, config.hidden_size, bias=False)
        self.classifier = nn.Linear(config.hidden_size , config.num_labels)
        self.mse_criterion = nn.MSELoss()
        self.axial_transformer = AxialTransformer_by_entity(emb_size = config.hidden_size, dropout=0.0, num_layers=6, heads=8)
        self.emb_size = emb_size
        self.threshold = nn.Threshold(0,0)
        self.block_size = block_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(args.dropout)
    
    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        sent_embs = []
        batch_entity_embs = []
        b, seq_l, h_size = sequence_output.size()
        #n_e = max([len(x) for x in entity_pos])
        n_e = 42
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            entity_lens = []

            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                     

                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            s_ne, _ = entity_embs.size()

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            
            pad_hs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_ts = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_hs[:s_ne, :s_ne, :] = hs.view(s_ne, s_ne, h_size)
            pad_ts[:s_ne, :s_ne, :] = ts.view(s_ne, s_ne, h_size)


            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            m = torch.nn.Threshold(0,0)
            ht_att = m((h_att * t_att).sum(1))
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-10)
            
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            pad_rs = torch.zeros((n_e, n_e, h_size)).to(sequence_output)
            pad_rs[:s_ne, :s_ne, :] = rs.view(s_ne, s_ne, h_size)
            hss.append(pad_hs)
            tss.append(pad_ts)
            rss.append(pad_rs)
            batch_entity_embs.append(entity_embs)
        hss = torch.stack(hss, dim=0)
        tss = torch.stack(tss, dim=0)
        rss = torch.stack(rss, dim=0)
        batch_entity_embs = torch.cat(batch_entity_embs, dim=0)
        return hss, rss, tss, batch_entity_embs

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                teacher_logits = None,
                negative_mask = None,
                Adj=None
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        bs, seq_len, h_size = sequence_output.size()
        bs, num_heads, seq_len, seq_len = attention.size()
        ctx_window = 300
        stride = 25
        device = sequence_output.device.index
        ne = 42
        nes = [len(x) for x in entity_pos]
        hs_e, rs_e, ts_e, batch_entity_embs = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs_e = torch.tanh(self.head_extractor(torch.cat([hs_e, rs_e], dim=3)))
        ts_e = torch.tanh(self.tail_extractor(torch.cat([ts_e, rs_e], dim=3)))       
    
        b1_e = hs_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        b2_e = ts_e.view(bs, ne, ne, self.emb_size // self.block_size, self.block_size)
        bl_e = (b1_e.unsqueeze(5) * b2_e.unsqueeze(4)).view(bs, ne, ne, self.emb_size * self.block_size)
        
        if negative_mask is not None:
            bl_e = bl_e * negative_mask.unsqueeze(-1)

        feature =  self.projection(bl_e)
        feature = self.axial_transformer(feature) + feature
        label_embeddings = []
                
        logits_c = self.classifier(feature)
        
        self_mask = (1 - torch.diag(torch.ones((ne)))).unsqueeze(0).unsqueeze(-1).to(sequence_output)
        logits_classifier = logits_c * self_mask
        logits_classifier = torch.cat([logits_classifier.clone()[x, :nes[x], :nes[x] , :].reshape(-1, self.config.num_labels) for x in range(len(nes))])

        output = (logits_classifier, self.loss_fnt.get_label(logits_classifier, num_labels=self.num_labels))
        
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(device)
            loss_classifier = self.loss_fnt(logits_classifier.view(-1, self.config.num_labels).float(), labels.float())
            output =  (loss_classifier, ) + output
            if teacher_logits is not None:
                teacher_logits = torch.cat(teacher_logits, dim=0).to(logits_classifier)
                mse_loss = self.mse_criterion(logits_classifier, teacher_logits)
                output = output + 1.0 *  mse_loss
        else:
            logits = logits_classifier.view(-1, self.config.num_labels)
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), logits)
                        
        return output


class TeacherEnhancer(nn.Module):
    def __init__(self, num_classes, embedding_dim, label_embedding_dim, hidden_dim, block_size=64):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_embedder = nn.Embedding(self.num_classes, label_embedding_dim)

        in_dim = embedding_dim * block_size + label_embedding_dim

        self.retain_label = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, y_n):
        y_emb = self.label_embedder(y_n).mean(dim=1)
        hin = torch.cat([features, y_emb], dim=-1)
        retain_logit = self.retain_label(hin)
        retain_conf = torch.sigmoid(retain_logit)
        return retain_conf