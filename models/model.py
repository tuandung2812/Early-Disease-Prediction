import torch
from torch import nn

from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.utils import DotProductAttention


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Model(nn.Module):
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, t_attention_size, t_output_size,
                 output_size, dropout_rate, activation):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 32)
        self.classifier = Classifier(hidden_size, output_size, dropout_rate, activation)


        self.prior_fc = nn.Linear(1,64)
        self.relu = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(64)

    def forward(self, code_x, divided, neighbors, lens, prior):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        # print('code_x',code_x.shape)
        i = 0
        # prior = prior.unsqueeze(1).repeat(1,64)
        prior = prior.unsqueeze(1)

        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
            i +=1
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            # print(prior.shape)

            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                # print(c_it.shape,n_it.shape)
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings, prior)
                # print(co_embeddings.shape)
                # print(no_embeddings.shape)

                # prior_out = self.prior_fc(prior)
                # # prior_out = self.relu(prior_out)
                # # prior_out = self.batchnorm(prior_out)

                # co_embeddings_with_prior = co_embeddings * prior_out
                # no_embeddings_with_prior = no_embeddings * prior_out

                # co_embeddings_with_prior = self.layernorm(co_embeddings_with_prior)
                # no_embeddings_with_prior = self.layernorm(no_embeddings_with_prior)

                # co_embeddings_with_prior = self.relu(co_embeddings_with_prior)
                # no_embeddings_with_prior = self.relu(no_embeddings_with_prior)

                # co_embeddings = co_embeddings + co_embeddings_with_prior
                # no_embeddings = no_embeddings + no_embeddings_with_prior
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                # print(output_it.shape)
                # print(output_it.shape)
                output_i.append(output_it)
            # print(torch.vstack(output_i).shape)
            output_i = self.attention(torch.vstack(output_i))
            # print(output_i.shape)
            output.append(output_i)
        # print(i)
        output = torch.vstack(output)
        output = self.classifier(output)
        # print(output)
        return output
