import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import math


from layers import HyperGraphAttentionLayerSparse, HGCN
from utils import build_hypergraph
import datetime

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len + 1, 1, d_model)
        pe[1:, 0, 0::2] = torch.sin(position * div_term)
        pe[1:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, sec_pos_label=None, in_sec_pos_label=None):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        if sec_pos_label is not None:
            x += self.pe[sec_pos_label.long().permute(1, 0), 0, :] * 1e-3
        if in_sec_pos_label is not None:
            x += self.pe[in_sec_pos_label.long().permute(1, 0), 0, :] * 1e-3
        return self.dropout(x).permute(1, 0, 2)


class BERT_Encoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased'):
        super(BERT_Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def forward(self, text, sec_sen_num):
        sen_nodes = []
        word_nodes = []
        word_nums = []
        # Encode text
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, max_length=128, truncation=True).to(device)
        input_ids = encoded_input["input_ids"]
        output = self.model(**encoded_input)["last_hidden_state"]

        for i in range(output.shape[0]):
            word_num = 0
            for j in range(output.shape[1]):
                if input_ids[i, j] == 101:
                    sen_nodes.append(output[i, j])
                elif input_ids[i, j] != 102 and input_ids[i, j] != 0:
                    word_nodes.append(output[i, j])
                    word_num += 1
            word_nums.append(word_num)

        nodes = sen_nodes + word_nodes
        features = torch.stack(nodes, dim=0)

        word_nums =torch.tensor(word_nums)
        sen_num = len(sen_nodes)
        edges = build_hypergraph(sec_sen_num, sen_num, word_nums).cuda()

        return features.unsqueeze(0), edges.unsqueeze(0), sen_num


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=True)


    def forward(self, x, H):
        x_res = x
        x = self.gat1(x, H) + x_res
        x = F.dropout(x, self.dropout, training=self.training)
        x_res = x
        x = self.gat2(x, H) + x_res
        return x


class Hyper_Atten_Block(nn.Module):
    def __init__(self, input_size, head_num, head_hid, dropout):
        super().__init__()
        self.dropout = dropout
        self.gat = nn.ModuleList( [HyperGraphAttentionLayerSparse(input_size, head_hid, dropout=self.dropout, alpha=0.2, transfer=True,
                                            concat=True) for _ in range(head_num)])

        self.ln = nn.LayerNorm([input_size])
        self.hm = nn.Sequential(nn.Linear(head_num * head_hid, input_size),
                                 nn.LeakyReLU(),
                                 nn.Dropout(self.dropout),
                                 )

        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.LayerNorm([input_size]),
            nn.Dropout(self.dropout),
        )

    def forward(self, x, H):
        x_res = x
        x = torch.concat([gat(x, H) for gat in self.gat], dim=-1)
        x = self.hm(x) + x_res
        x = self.ln(x)
        x = self.ffn(x) + x
        x = self.ln(x)
        return x


class HGNN_ATT_MH(nn.Module):
    def __init__(self, input_size, head_num, head_hid, output_size, layers=2, dropout=0.3):
        super(HGNN_ATT_MH, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.hatt = nn.ModuleList([Hyper_Atten_Block(input_size, head_num, head_hid, dropout) for _ in range(layers)])


    def forward(self, x, H):
        for att in self.hatt:
            x = att(x, H) + x
        return x

class HGraph_Sum(nn.Module):
    def __init__(self, input_size, n_hid, dropout=0.3, g_layer=2):
        super(HGraph_Sum, self).__init__()
        self.dropout = dropout
        self.position = PositionalEncoding(input_size)
        self.hgraph_layer = HGNN_ATT_MH(1024, head_num=8, head_hid=128, output_size=1024, dropout=dropout, layers=g_layer)
        self.ln = nn.LayerNorm([1024])
        self.input_layer = nn.Linear(input_size, 1024)
        self.output_layer = nn.Linear(1024, n_hid)
        self.final_layer = nn.Linear(n_hid, 1)

    def forward(self, feature, edges, sen_num, mask, sec_pos_label=None, in_sec_pos_label=None, pos_label=None):
        feature = self.position(feature, sec_pos_label, in_sec_pos_label)
        feature = F.leaky_relu(self.input_layer(feature))

        feature = self.hgraph_layer(feature, edges)

        feature = F.leaky_relu(self.output_layer(feature))
        feature = F.dropout(feature, self.dropout, training=self.training)
        feature = self.final_layer(feature)

        if not self.training:
            feature = torch.where(mask==1, feature.double(), -1e3)
        return feature

