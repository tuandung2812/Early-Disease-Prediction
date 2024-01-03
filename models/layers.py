import torch
from torch import nn

from models.utils import SingleHeadAttentionLayer
import numpy as np

class EmbeddingLayer(nn.Module):
    def __init__(self, code_num, code_size, graph_size):
        super().__init__()
        self.code_num = code_num
        self.c_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.n_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, code_size)))
        self.u_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, graph_size)))

        if code_num == 6743:
        # try:
            bert_embeddings = torch.from_numpy(np.load('data/mimic4/bert_umap_embeddings.npy'))
        else:
            bert_embeddings = torch.from_numpy(np.load('data/mimic3/bert_umap_embeddings.npy'))

        self.c_embeddings.data = bert_embeddings
        self.n_embeddings.data = bert_embeddings
        self.u_embeddings.data = bert_embeddings
        # except:
        #     bert_embeddings = torch.from_numpy(np.load('data/mimic3/bert_umap_embeddings.npy'))
        #     self.c_embeddings.data = bert_embeddings
        #     self.n_embeddings.data = bert_embeddings
        #     self.u_embeddings.data = bert_embeddings

    def forward(self):
        return self.c_embeddings, self.n_embeddings, self.u_embeddings


class GraphLayer(nn.Module):
    def __init__(self, adj, code_size, graph_size):
        super().__init__()
        self.adj = adj
        self.dense = nn.Linear(code_size, graph_size)
        self.activation = nn.LeakyReLU()

    def forward(self, code_x, neighbor, c_embeddings, n_embeddings,prior):
        # print(code_x.shape, neighbor.shape)
        center_codes = torch.unsqueeze(code_x, dim=-1)
        # print('center_codes',center_codes,center_codes.shape)
        neighbor_codes = torch.unsqueeze(neighbor, dim=-1)
        # print('neighbor_codes',neighbor_codes,neighbor_codes.shape)
        # print(center_codes.shape,c_embeddings.shape)
        # print(center_codes.shape, c_embeddings.shape)
        center_embeddings = center_codes * c_embeddings
        # print('center_embeddings',center_embeddings)
        neighbor_embeddings = neighbor_codes * n_embeddings

        # print(center_embeddings.shape, neighbor_embeddings.shape)
        # print(prior.shape)
        # prior = prior.unsqueeze(1).repeat(1,64)
        # center_embeddings = center_embeddings * prior
        # neighbor_embeddings = neighbor_embeddings * prior
        cc_embeddings = center_codes * torch.matmul(self.adj, center_embeddings)
        # print(cc_embeddings)
        cn_embeddings = center_codes * torch.matmul(self.adj, neighbor_embeddings)
        # print(cn_embeddings)
        nn_embeddings = neighbor_codes * torch.matmul(self.adj, neighbor_embeddings)
        nc_embeddings = neighbor_codes * torch.matmul(self.adj, center_embeddings)
        # print(center_embeddings.shape)
        co_embeddings = self.activation(self.dense(center_embeddings + cc_embeddings + cn_embeddings))
        # print(co_embeddings.shape)
        no_embeddings = self.activation(self.dense(neighbor_embeddings + nn_embeddings + nc_embeddings))
        # print('co_embeddings',co_embeddings,co_embeddings.shape)
        return co_embeddings, no_embeddings


class TransitionLayer(nn.Module):
    def __init__(self, code_num, graph_size, hidden_size, t_attention_size, t_output_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size=graph_size, hidden_size=hidden_size)
        self.single_head_attention = SingleHeadAttentionLayer(graph_size, graph_size, t_output_size, t_attention_size)
        self.activation = nn.Tanh()

        self.code_num = code_num
        self.hidden_size = hidden_size

    def forward(self, t, co_embeddings, divided, no_embeddings, unrelated_embeddings, hidden_state=None):
        m1, m2, m3 = divided[:, 0], divided[:, 1], divided[:, 2]
        m1_index = torch.where(m1 > 0)[0]
        m2_index = torch.where(m2 > 0)[0]
        m3_index = torch.where(m3 > 0)[0]
        h_new = torch.zeros((self.code_num, self.hidden_size), dtype=co_embeddings.dtype).to(co_embeddings.device)
        output_m1 = 0
        output_m23 = 0
        if len(m1_index) > 0:
            m1_embedding = co_embeddings[m1_index]
            h = hidden_state[m1_index] if hidden_state is not None else None
            h_m1 = self.gru(m1_embedding, h)
            h_new[m1_index] = h_m1
            output_m1, _ = torch.max(h_m1, dim=-2)
        if t > 0 and len(m2_index) + len(m3_index) > 0:
            q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
            v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]])
            h_m23 = self.activation(self.single_head_attention(q, q, v))
            h_new[m2_index] = h_m23[:len(m2_index)]
            h_new[m3_index] = h_m23[len(m2_index):]
            output_m23, _ = torch.max(h_m23, dim=-2)
        if len(m1_index) == 0:
            output = output_m23
        elif len(m2_index) + len(m3_index) == 0:
            output = output_m1
        else:
            output, _ = torch.max(torch.vstack([output_m1, output_m23]), dim=-2)
        return output, h_new

