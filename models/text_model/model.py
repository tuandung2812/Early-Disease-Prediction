import torch
from torch import nn

from models.text_model.layers import EmbeddingLayer, GraphLayer, TransitionLayer
from models.text_model.utils import DotProductAttention, SingleHeadAttentionLayer, CustomAttentionLayer
from models.text_model.text_transformer import NMT_tran


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
                 output_size, dropout_rate, activation,
                 transformer_hidden_size = 768,
                 transformer_att_head_num = 4,text_embedding_size  = 128,vocab_size = 5000,
                 encoder_layers=4,transformer_dropout_rate=0.2):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(hidden_size, 32)

        self.text_transformer = NMT_tran(transformer_hidden_size,
                 transformer_att_head_num ,text_embedding_size ,vocab_size ,
                 encoder_layers,transformer_dropout_rate)
        
        self.icd_text_attention = CustomAttentionLayer(query_size=hidden_size,key_size=transformer_hidden_size,value_size=1,attention_size=64)
        self.classifier = Classifier(hidden_size + transformer_hidden_size, output_size, dropout_rate, activation)

    def forward(self, code_x, divided, neighbors, lens, note, note_mask):
        embeddings = self.embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        output = []
        # print('code_x',code_x.shape)
        i = 0
        for code_x_i, divided_i, neighbor_i, len_i in zip(code_x, divided, neighbors, lens):
            i +=1
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i))):
                # print(c_it.shape,n_it.shape)
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                output_it, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                # print(output_it.shape)
                output_i.append(output_it)
            # print(torch.vstack(output_i).shape)
            output_i = self.attention(torch.vstack(output_i))
            # print(output_i.shape)
            output.append(output_i)
        # print(i)
        icd_output = torch.vstack(output)
        # print(icd_output.shape)

        # text_embedding = self.text_transformer(note,note_mask,mode='calc_only_text')
        all_text_hiddens, first_hidden = self.text_transformer.encode(note,note_mask)
        # print(icd_output.shape,all_text_hiddens.shape)
        icd_text_att = self.icd_text_attention(queries=icd_output,keys=all_text_hiddens,values=all_text_hiddens,mask=note_mask)
        output = torch.cat([icd_output,icd_text_att],dim=-1) 
        # print(all_text_hiddens.shape)
        # print(first_hidden.shape)
        # output = output.squeeze(1)
        # print(output.shape)
        output = self.classifier(output)
        # print(output)
        return output
