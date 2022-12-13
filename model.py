import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable
import utils as utils

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, embed_d, num_heads,dropout, mode, params):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_d = embed_d
        self.layer_dim = layer_dim
        self.hasConcept = params.hasConcept
        self.output_dim = output_dim
        self.mode = mode
        self.p_lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        if mode == 2 or self.hasConcept == 0:
            self.y_modify = nn.Linear(embed_d * num_heads * 4, embed_d * 2, bias=True)
            self.x_modify = nn.Linear(embed_d * num_heads * 3, embed_d * 2, bias=True)
        else:
            self.y_modify = nn.Linear(embed_d * num_heads * 8, embed_d * 2, bias=True)
            self.x_modify = nn.Linear(embed_d * num_heads * 6, embed_d * 2, bias=True)
        self.fc1 = nn.Linear(self.embed_d * 2 + self.hidden_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, 1, bias=True)
        self.rel = nn.ReLU()

        self.n_kc = params.n_knowledge_concept + 1
        self.n_exercise = params.n_exercise + 1
        # self.difficult_param = nn.Embedding(self.n_pid, 1)
        self.gpu = params.gpu

        self.masked_attn_head_exercise = MultiHeadAttention_exercise(self.n_kc, self.n_exercise, self.hasConcept, embed_d, num_heads, mode, self.gpu)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal(self.x_modify.weight)
        nn.init.kaiming_normal(self.y_modify.weight)
        nn.init.kaiming_normal(self.fc1.weight)
        nn.init.kaiming_normal(self.fc2.weight)
        nn.init.constant(self.x_modify.bias, 0)
        nn.init.constant(self.y_modify.bias, 0)
        nn.init.constant(self.fc1.bias, 0)
        nn.init.constant(self.fc2.bias, 0)

    def forward(self, q_seq, qa_seq, pid_seq, target):
        p_query, p_masked_key, p_value= self.masked_attn_head_exercise(q_seq, pid_seq)

        p_attention_scores = torch.matmul(p_query.unsqueeze(-2), p_masked_key.transpose(-1, -2)).squeeze(2)
        p_attention_scores = p_attention_scores / math.sqrt(self.embed_d)
        mask = (p_attention_scores == 0.0).bool()
        p_attention_scores = p_attention_scores.masked_fill(mask, -np.inf)
        p_attention_wights = nn.Softmax(dim=-1)(p_attention_scores)
        p_attention_wights = torch.where(torch.isnan(p_attention_wights), torch.full_like(p_attention_wights, 0.0),
                                         p_attention_wights)
        p_attention_embed = torch.mul(p_attention_wights.unsqueeze(-1), p_value.unsqueeze(1))

        zeros = torch.zeros_like(p_attention_embed)
        cat1 = torch.cat((zeros, p_attention_embed), -1)
        cat2 = torch.cat((p_attention_embed, zeros), -1)
        p_embed_data = torch.cat((cat1, cat2), -2)
        index = qa_seq * qa_seq.size(-1) + utils.varible(torch.arange(0, qa_seq.size(-1), 1), self.gpu)
        batch_size = index.size(0)
        index_i = index.chunk(batch_size, 0)
        batch_data = []
        for i, data_i in enumerate(p_embed_data.chunk(batch_size, 0)):
            result = torch.index_select(data_i, dim=-2, index=index_i[i].squeeze(0))
            batch_data.append(result)
        p_historical_relevance = torch.cat(batch_data, 0)
        p_disttotal_scores = torch.sum(p_historical_relevance, dim=-2,
                                       keepdim=False)

        # 全零向量拼接，答对拼右边，答错拼左边
        zeros = torch.zeros_like(p_value)
        cat1 = torch.cat((zeros, p_value), -1)
        cat2 = torch.cat((p_value, zeros), -1)
        embed_data = torch.cat((cat1, cat2), -2)
        location = qa_seq * qa_seq.size(-1) + utils.varible(torch.arange(0, qa_seq.size(-1), 1), self.gpu)
        batch_size = location.size(0)
        location_i = location.chunk(batch_size, 0)
        batch_problem_answer = []
        for i, data_i in enumerate(embed_data.chunk(batch_size, 0)):
            data = torch.index_select(data_i, dim=-2, index=location_i[i].squeeze(0))
            batch_problem_answer.append(data)
        p_input = torch.cat(batch_problem_answer, 0)
        p_input = torch.cat((p_input,p_disttotal_scores),-1)
        y_modify = self.y_modify(p_input)


        h0 = utils.varible(Variable(torch.zeros(self.layer_dim, p_value.size(0), self.hidden_dim)), self.gpu)
        c0 = utils.varible(Variable(torch.zeros(self.layer_dim, p_value.size(0), self.hidden_dim)), self.gpu)
        p_out, (hn, cn) = self.p_lstm(y_modify, (h0, c0))
        batch, maxstep, hidden_layer = p_out.shape
        p_out = p_out[:, : maxstep - 1, :]

        p_disttotal_scores = p_disttotal_scores[:, 1:, :]
        p_value = p_value[:, 1:, :]
        p_value = torch.cat((p_value,p_disttotal_scores),-1)
        x_modify = self.x_modify(p_value)


        prediction = torch.cat((p_out,x_modify), -1)
        y = self.rel(self.fc1(prediction))
        res = self.fc2(y)


        target_1d = target  # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)  # [batch_size * seq_len, 1]
        pred_1d = res.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        predict_loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return predict_loss, torch.sigmoid(filtered_pred), filtered_target


def mask_operation(seq, gpu):
    batch_size, max_step, embeded_num = seq.size()
    mask = 1 -  utils.varible(torch.triu(torch.ones((max_step, max_step), dtype=torch.float32)), gpu)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, embeded_num)
    sequence_masked = torch.mul(torch.unsqueeze(seq, dim=1), mask)
    return sequence_masked

class MultiHeadAttention_exercise(nn.Module):
    def __init__(self, n_kc, n_exercise, hasConcept, embed_d, num_heads, mode, gpu):
        super().__init__()
        self.mode = mode
        self.gpu = gpu
        self.n_kc = n_kc
        self.n_exercise = n_exercise
        self.hasConcept = hasConcept
        if hasConcept == 1:
            self.kc_embed = nn.Embedding(n_kc, embed_d, padding_idx=0)
        self.exercise_embed = nn.Embedding(n_exercise, embed_d, padding_idx=0)
        self.num_heads = num_heads
        self.E = nn.Linear(embed_d, embed_d, bias=True)

        if hasConcept == 0:
            self.linear_q = nn.Linear(embed_d, embed_d * num_heads, bias=True)
            self.linear_k = nn.Linear(embed_d, embed_d * num_heads, bias=True)
            self.linear_v = nn.Linear(embed_d, embed_d * num_heads, bias=True)
        else:
            if mode == 1 or mode == 3:
                self.linear_q = nn.Linear(embed_d * 2, embed_d * num_heads * 2, bias=True)
                self.linear_k = nn.Linear(embed_d * 2, embed_d * num_heads * 2, bias=True)
                self.linear_v = nn.Linear(embed_d * 2, embed_d * num_heads * 2, bias=True)
            elif mode == 2:
                self.linear_q = nn.Linear(embed_d, embed_d * num_heads, bias=True)
                self.linear_k = nn.Linear(embed_d, embed_d * num_heads, bias=True)
                self.linear_v = nn.Linear(embed_d, embed_d * num_heads, bias=True)
        self.init_params()

    def init_params(self):
        if self.hasConcept == 1:
            nn.init.kaiming_normal(self.kc_embed.weight)
        nn.init.kaiming_normal(self.exercise_embed.weight)
        nn.init.kaiming_normal(self.linear_q.weight)
        nn.init.kaiming_normal(self.linear_k.weight)
        nn.init.kaiming_normal(self.linear_v.weight)
        nn.init.constant(self.linear_q.bias, 0)
        nn.init.constant(self.linear_k.bias, 0)
        nn.init.constant(self.linear_v.bias, 0)

    def forward(self, kc_seq, exercise_seq):
        if self.hasConcept == 1:
            kc_embedd = self.kc_embed(kc_seq)
        exercise_embedd = self.exercise_embed(exercise_seq)

        if self.hasConcept == 1:
            if self.mode == 1:
                exercise_embedd = torch.cat((kc_embedd, exercise_embedd), -1)
            elif self.mode == 2:
                exercise_embedd = self.E(exercise_embedd)
                exercise_embedd = kc_embedd * exercise_embedd
            elif self.mode == 3:
                exercise_embedd = self.E(exercise_embedd)
                exercise_embedd = kc_embedd * exercise_embedd
                exercise_embedd = torch.cat((kc_embedd, exercise_embedd), -1)

        p_value = self.linear_v(exercise_embedd)
        p_query = self.linear_q(exercise_embedd)
        p_key = self.linear_k(exercise_embedd)
        p_masked_key = mask_operation(p_key, self.gpu)
        return  p_query, p_masked_key, p_value

