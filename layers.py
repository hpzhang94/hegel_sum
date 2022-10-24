import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class HyperGraphAttentionLayerSparse(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.transfer = transfer
        self.edge_node_fusion_layer = nn.Linear(2 * out_features, out_features)

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # number of edge
        N2 = adj.shape[2]  # number of node

        pair = adj.nonzero().t()

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])
        # print("x1:{}".format(x1.shape))

        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)
        # print("q1:{}".format(q1.shape))

        pair_h = torch.cat((q1, x1), dim=-1)
        # print("pair_h:{}".format(pair_h.shape))
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        # print("pair_e:{}".format(pair_e.shape))
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)
        # print("attention_edge:{}".format(attention_edge[0, 0]))

        edge = torch.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)

        edge_4att = edge.matmul(self.weight3)

        get = lambda i: edge_4att[i][adj[i].nonzero().t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        get = lambda i: x_4att[i][adj[i].nonzero().t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        assert not torch.isnan(pair_e).any()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)

        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)
        # print("attention_node:{}".format(attention_node[0, 0]))

        edge_feature = torch.matmul(attention_node, edge)
        node = torch.cat((edge_feature, x), dim=-1)
        node = F.dropout(self.edge_node_fusion_layer(node), self.dropout, training=self.training)

        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.W_Q = nn.Parameter(torch.zeros(size=(in_dim, hid_dim)))
        self.W_K = nn.Parameter(torch.zeros(size=(in_dim, hid_dim)))
        nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_K.data, gain=1.414)

    def forward(self, Q, K, mask=None):

        KW_K = torch.matmul(K, self.W_K)
        QW_Q = torch.matmul(Q, self.W_Q)
        if len(KW_K.shape) == 3:
            KW_K = KW_K.permute(0, 2, 1)
        elif len(KW_K.shape) == 4:
            KW_K = KW_K.permute(0, 1, 3, 2)
        att_w = torch.matmul(QW_Q, KW_K).squeeze(1)

        if mask is not None:
            att_w = torch.where(mask == 1, att_w.double(), float(-1e10))

        att_w = F.softmax(att_w / torch.sqrt(torch.tensor(Q.shape[-1])), dim=-1)
        return att_w

class HGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, drop_out):
        super(HGCN, self).__init__()
        self.edge_att = Attention(in_dim, hid_dim)
        self.node_att = Attention(in_dim, hid_dim)
        self.edge_linear_layer = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, adj):
        edge_fea = torch.matmul(adj, x) / (torch.sum(adj, dim=-1).unsqueeze(-1) + 1e-5)
        edge_fea = self.dropout(F.leaky_relu(self.edge_linear_layer(edge_fea))) #(batch, edge_num, fea_dim)

        att_e = self.edge_att(edge_fea, x, adj) #(batch, edge_num, node_num)
        edge_fea = torch.matmul(att_e.float(), x.float()) #(batch, edge_num, fea_dim)
        att_n = self.node_att(x, edge_fea)
        x = torch.matmul(att_n.float(), edge_fea.float())
        return F.leaky_relu(x)