import os
from random import random

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import math
import copy

import settings

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


# 对每个签到进行嵌入
# Transformer 直接用的 nn.Embedding
class TransRecEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class CatPreEmbedding(nn.Module):  # c h t 用户类别签到记录，用来挖掘用户的类别偏好
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        cat_num = vocab_size["cat"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        cat_emb = self.cat_embed(x[1])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])
        return torch.cat((cat_emb, hour_emb, day_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split the embedding into self.heads different pieces
        # Multi head
        # [len, embed_size] --> [len, heads, head_dim]
        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        # 爱因斯坦求和约定，矩阵计算的简单表示方式
        energy = torch.einsum("qhd,khd->hqk", [queries, keys])  # [heads, query_len, key_len]

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)  # [heads, query_len, key_len]

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )  # [query_len, key_len]

        out = self.fc_out(out)  # [query_len, key_len]

        return out


# 对应 TransformerBlock
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# 对应 TransformerEncoder
class TransRecEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,  # TransRecEmbedding
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransRecEncoder, self).__init__()

        # Transformer 直接用的 nn.Embedding
        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        # num_encoder_layers 个 EncoderBlock
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        # 因为是在 Encoder 中，所以 value, key, query 都一样
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# 因为 query 和 key 的维度不一样大，所以专门写了这个 Attention
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # 将 q 的维度调整为和 k 一样大
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):  # q[embed_size]
        q = self.expansion(query)  # q=[embed_size*5] #一维
        weight = torch.softmax(torch.inner(q, key), dim=0)  # [len]
        weight = torch.unsqueeze(weight, 1)  # [len, 1]
        out = torch.sum(torch.mul(value, weight), 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


# 推荐模型
class TransRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=2,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.1,
            random_mask=True,
            mask_prop=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.random_mask = random_mask
        self.mask_prop = mask_prop

        # LAYERS 每个签到的初始化嵌入
        self.embedding = TransRecEmbedding(
            f_embed_size,
            vocab_size
        )

        if settings.LS_strategy == 'TransLSTM':
            self.long_lstm = nn.LSTM(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
                num_layers=num_lstm_layers,
                dropout=0
            )
            self.short_lstm = nn.LSTM(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
                num_layers=num_lstm_layers,
                dropout=0
            )

        elif settings.LS_strategy == 'DoubleTrans':
            self.short_encoder = TransRecEncoder(
                self.embedding,
                self.total_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )
            self.long_encoder = TransRecEncoder(
                self.embedding,
                self.total_embed_size,
                num_encoder_layers,
                num_heads,
                forward_expansion,
                dropout_p,
            )
        else:
            pass

        if settings.enable_catpre_embedding:
            # LAYERS 每个类别签到的初始化嵌入
            self.car_pre_embedding = CatPreEmbedding(
                f_embed_size,
                vocab_size
            )

            self.cat_pre_lstm = nn.LSTM(
                input_size=f_embed_size * 3,
                hidden_size=f_embed_size * 3,
                num_layers=num_lstm_layers,
                dropout=0
            )

            self.final_attention = Attention(
                qdim=f_embed_size * 3,
                kdim=self.total_embed_size
            )

            self.cat_attention = Attention(
                qdim=f_embed_size * 1,
                kdim=self.total_embed_size
            )

            self.time_attention = Attention(  # 时间注意力机制
                qdim=f_embed_size * 3,
                kdim=f_embed_size * 3
            )

        else:
            self.final_attention = Attention(
                qdim=f_embed_size,
                kdim=self.total_embed_size
            )

        self.alpha_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )

        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        if settings.net_init:
            self.out_linear.apply(self.init_add_weight)

        self.loss_func = nn.CrossEntropyLoss()

        if settings.enable_alpha:
            self.alpha_gru = nn.GRU(
                input_size=self.total_embed_size,
                hidden_size=self.total_embed_size,
            )

            if settings.enable_catpre_embedding:
                self.alpha_input_embed_size = 3 * self.total_embed_size + 5 * f_embed_size
            else:
                self.alpha_input_embed_size = 3 * self.total_embed_size + 2 * f_embed_size
            self.alpha_linear = nn.Sequential(
                nn.Linear(self.alpha_input_embed_size, self.total_embed_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(self.total_embed_size, f_embed_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(f_embed_size, 1),
                nn.Sigmoid())
            self.alpha = 0.5

        # 使用一个 dict 存储需要追踪的参数
        self.track_parameters = {}

    #  定义一个网络初始化函数
    def init_add_weight(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == nn.Embedding:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # 对长期轨迹进行掩码
    def randomly_mask(self, sequences, mask_prop):  # 随机掩码
        sample_list = []
        for seq in sequences:  # each long term sequences
            seq_len = len(seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            # mask long term sequences' POI, cat, hour, day
            seq[0, masked_index] = self.vocab_size["POI"]
            seq[1, masked_index] = self.vocab_size["cat"]
            # 2 user 不能 mask
            seq[3, masked_index] = self.vocab_size["hour"]
            seq[4, masked_index] = self.vocab_size["day"]

            sample_list.append(seq)

        return sample_list

    # 对短期轨迹进行掩码
    def randomly_mask_short(self, sequence, mask_prop):  # 随机掩码
        seq = sequence
        seq_len = len(seq[0])
        mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
        masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
        masked_index = masked_index[:mask_count]  # randomly generate mask index

        # mask long term sequences' POI, cat, hour, day
        seq[0, masked_index] = self.vocab_size["POI"]
        seq[1, masked_index] = self.vocab_size["cat"]
        # 2 user 不能 mask
        seq[3, masked_index] = self.vocab_size["hour"]
        seq[4, masked_index] = self.vocab_size["day"]

        return seq

    # 对类别轨迹进行随机掩码
    def randomly_mask_cat(self, sequences, cat_mask_prop=0.1):  # 随机掩码
        sample_list = []
        for seq in sequences:  # each long term sequences
            seq_len = len(seq[0])
            mask_count = torch.ceil(cat_mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            # mask long term sequences'cat, hour, day
            seq[1, masked_index] = self.vocab_size["cat"]
            # 2 user 不能 mask
            seq[3, masked_index] = self.vocab_size["hour"]
            seq[4, masked_index] = self.vocab_size["day"]

            sample_list.append(seq)

        return sample_list

    def cal_CL_loss(self, short, long, short_proxy, long_proxy):
        def euclidean_distance(x, y):
            return torch.square(x - y)

        if settings.CL_strategy == 'BPR':
            long_mean_recent_loss = torch.sum(F.softplus(torch.sum(long * (-long_proxy + short_proxy), dim=-1)))
            short_recent_mean_loss = torch.sum(F.softplus(torch.sum(short * (-short_proxy + long_proxy), dim=-1)))
            mean_long_short_loss = torch.sum(F.softplus(torch.sum(long_proxy * (-long + short), dim=-1)))
            recent_short_long_loss = torch.sum(F.softplus(torch.sum(short_proxy * (-short + long), dim=-1)))
            return long_mean_recent_loss + short_recent_mean_loss + mean_long_short_loss + recent_short_long_loss
        elif settings.CL_strategy == 'Triplet':
            triplet_loss = (
                nn.TripletMarginWithDistanceLoss(distance_function=euclidean_distance, margin=1.0, reduction='sum'))
            long_loss_1 = triplet_loss(long, long_proxy, short_proxy)
            long_loss_2 = triplet_loss(long_proxy, long, short)
            short_loss_1 = triplet_loss(short, short_proxy, long_proxy)
            short_loss_2 = triplet_loss(short_proxy, short, long)
            return (long_loss_1 + long_loss_2 + short_loss_1 + short_loss_2) * 0.001
        elif settings.CL_strategy == 'NativeNCE':  # infoNCE
            pos1 = torch.mean(torch.mul(long, long_proxy))
            pos2 = torch.mean(torch.mul(short, short_proxy))
            neg1 = torch.mean(torch.mul(long, short_proxy))
            neg2 = torch.mean(torch.mul(short, long_proxy))
            pos = (pos1 + pos2) / 2
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss
        elif settings.CL_strategy == 'CosineNCE':
            pos1 = F.cosine_similarity(long, long_proxy, dim=0)
            pos2 = F.cosine_similarity(short, short_proxy, dim=0)
            neg1 = F.cosine_similarity(long, short_proxy, dim=0)
            neg2 = F.cosine_similarity(short, long_proxy, dim=0)
            pos = (pos1 + pos2) / 2
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss
        else:
            raise NotImplementedError

    def forward(self, sample, is_train=True):
        # process input sample
        # [(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq])],[(seq2)],...]
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]

        if settings.enable_drop:
            short_term_features = short_term_sequence[:, :- 1 - settings.drop_steps]
            target = short_term_sequence[0, -1 - settings.drop_steps]
        else:
            short_term_features = short_term_sequence[:, :- 1]
            target = short_term_sequence[0, -1]

        user_id = short_term_sequence[2, 0]
        last_time = short_term_features[3, -1]  # t+1时刻
        last_day = short_term_features[4, 0]  # 当前day信息

        copied_long_term_sequences = copy.deepcopy(long_term_sequences)
        copied_short_term_sequence = copy.deepcopy(short_term_features)

        # random mask input sequences
        if self.random_mask:
            copied_long_term_sequences = self.randomly_mask(copied_long_term_sequences, self.mask_prop)

        if settings.enable_random_mask_short:
            copied_short_term_sequence = self.randomly_mask_short(copied_short_term_sequence, mask_prop=0.1)

        # region long-term
        if settings.LS_strategy == 'TransLSTM':
            long_term_out = []  # [6*10, 8*10, ...]
            for seq in copied_long_term_sequences:
                long_embedding = self.embedding(seq)  # 嵌入层
                output, _ = self.long_lstm(torch.unsqueeze(long_embedding, 0))
                long_output = torch.squeeze(output)
                long_term_out.append(long_output)  # [seq_num, len, emb_size]
            long_term_catted = torch.cat(long_term_out, dim=0)
        else:
            long_term_out = []  # [6*10, 8*10, ...]
            for seq in copied_long_term_sequences:
                output = self.long_encoder(feature_seq=seq)
                long_term_out.append(output)  # [seq_num, len, emb_size]
            long_term_catted = torch.cat(long_term_out, dim=0)
        # endregion

        # region short-term
        short_embedding = self.embedding(copied_short_term_sequence)
        if settings.LS_strategy == 'TransLSTM':
            output, _ = self.short_lstm(torch.unsqueeze(short_embedding, 0))
            short_term_state = torch.squeeze(output)
        elif settings.LS_strategy == 'DoubleTrans':
            short_term_state = self.short_encoder(feature_seq=copied_short_term_sequence)
        else:
            raise NotImplementedError
        # endregion

        # region user_embedding
        if settings.enable_catpre_embedding:
            # 类别偏好  改成LSTM方式获取
            # region 版本一的代码
            user_seqs = long_term_sequences  # 二维tensor[5=特征个数,len=序列长度]
            if settings.enable_cat_pre_mask:
                user_seqs = self.randomly_mask_cat(user_seqs, cat_mask_prop=0.1)

            cat_pre_out = []
            for seq in user_seqs:
                cat_embedding = self.car_pre_embedding(seq)  # 嵌入层
                output, _ = self.cat_pre_lstm(torch.unsqueeze(cat_embedding, 0))
                cat_out = torch.squeeze(output)
                cat_pre_out.append(cat_out)  # [seq_num, len, emb_size]

            s_cat_embedding = self.car_pre_embedding(short_term_features)  # 嵌入层
            s_output, _ = self.cat_pre_lstm(torch.unsqueeze(s_cat_embedding, 0))
            s_cat_out = torch.squeeze(s_output)
            cat_pre_out.append(s_cat_out)  # [seq_num, len, emb_size*3]
            cat_pre = torch.mean(torch.cat(cat_pre_out, dim=0), dim=0)  # 获取到cat感知 理想状态一维[embed_size*3]
            # 扩充维度，变成[1,embed_size*3]
            cat_pre = cat_pre.unsqueeze(0)

            # 加入时间注意力机制，将当前时间（hour+day）作为查询健   如果跑起来没有效果就改成下一个时刻的类别签到序列
            next_cat = short_term_sequence[1, -1]
            next_time = short_term_sequence[3, -1]  # ,tn+1时刻
            next_day = short_term_features[4, 0]  # 当前day信息

            next_cat_embedding = self.embedding.cat_embed(next_cat)
            next_time_embedding = self.embedding.hour_embed(next_time)  # 一维tensor[embed_size]
            next_day_embedding = self.embedding.day_embed(next_day)  # 一维tensor[embed_size]

            time_embedding = torch.cat((next_cat_embedding, next_time_embedding, next_day_embedding),
                                       dim=-1)  # 一维[embed_size*3]
            time_embedding = time_embedding.unsqueeze(0)

            # 下一个poi访问 类别的嵌入作为查询健
            cat_pre_embed = self.time_attention(time_embedding, cat_pre, cat_pre)  # 二维 tensor[1,embed_size*3]
            # 压缩维度 ,必须压缩了才能进行后面的偏好增强操作 一维[embed_size*3]
            cat_pre_embed = cat_pre_embed.squeeze()
        # endregion

        if settings.enable_catpre_embedding:
            long_term_prefer = self.final_attention(cat_pre_embed, long_term_catted,
                                                    long_term_catted)  # 一维tensor[embed_size*5] 偏好序列进行注意力机制融合
            short_term_prefer = self.final_attention(cat_pre_embed, short_term_state,
                                                     short_term_state)  # 一维tensor[embed_size*5]
        else:
            long_term_prefer = torch.mean(long_term_catted, dim=0)  # 理论上应该保持一维[embed_size*5]
            short_term_prefer = torch.mean(short_term_state, dim=0)  # 理论上应该保持一维[embed_size*5]

        if settings.enable_CL:
            # proxy for long-term
            long_term_embeddings = []
            for seq in long_term_sequences:
                seq_embedding = self.embedding(seq)
                long_term_embeddings.append(seq_embedding)
            long_term_embeddings = torch.cat(long_term_embeddings, dim=0)
            long_term_proxy = torch.mean(long_term_embeddings, dim=0)
            # proxy for short-term
            short_term_proxy = torch.mean(short_embedding, dim=0)

            CL_loss = self.cal_CL_loss(short_term_prefer, long_term_prefer,
                                       short_term_proxy, long_term_proxy)

        if not is_train:  # 将测试集的长短期偏好存储起来
            # 这两句话无论放在 CL_loss 前面还是后面都是一样的结果
            settings.long_term_preference_list.append(long_term_prefer.cpu().detach())
            settings.short_term_preference_list.append(short_term_prefer.cpu().detach())

        # final output
        if settings.enable_alpha:
            # region fusion long and short like CLSR
            all_term_features = torch.cat([torch.cat(long_term_sequences, dim=-1), short_term_features],
                                          dim=-1)  # 二维tensor[5=特征个数,len=序列长度]
            all_term_embedding = self.embedding(all_term_features)  # 二维[len=序列长度，300=总嵌入大小]
            _, h_n = self.alpha_gru(all_term_embedding)  # 二维tensor[1,300]
            h_n = torch.squeeze(h_n)  # 维度压缩，一维tensor[300]
            last_time_embedding = self.embedding.hour_embed(last_time)  # 一维tensor[60]
            last_day_embedding = self.embedding.day_embed(last_day)  # 一维tensor[embed_size]
            if settings.enable_catpre_embedding:

                concat_all = torch.cat(
                    [h_n, long_term_prefer, short_term_prefer, cat_pre_embed, last_time_embedding, last_day_embedding],
                    dim=-1)  # 一维tensor[=3*total_embed_size+embed_size*3+1*embed_size]
            else:
                concat_all = torch.cat(
                    [h_n, long_term_prefer, short_term_prefer, last_time_embedding, last_day_embedding],
                    dim=-1)  # 一维tensor[3*total_embed_size+1*embed_size]
            self.alpha = self.alpha_linear(concat_all)  # 一维tensor[长期偏好融合值]
            self.track_parameters['alpha'] = self.alpha.item()
            # alpha 放到 long
            final_att = long_term_prefer * self.alpha + short_term_prefer * (1 - self.alpha)  # 一维tensor[300]
            output = self.out_linear(final_att)  # 一维tensor[POI个数]
            # endregion
        else:
            if settings.enable_filatt:  # 以注意力的形式融合长短期偏好
                if settings.enable_CL:
                    long_term_prefer = long_term_prefer.unsqueeze(0)  # [1,embed_size*5]
                    short_term_prefer = short_term_prefer.unsqueeze(0)  # [1,embed_size*5]
                    h = torch.cat((long_term_prefer, short_term_prefer), dim=0)  # 二维[2,embed_size*5]
                    f_user_embed = self.embedding.user_embed(user_id)  # 一维[embed_size]
                    final_att = self.alpha_attention(f_user_embed, h, h)
                elif settings.enable_catpre_embedding:
                    h = torch.cat((long_term_catted, short_term_state))  # concat long and short 二维tensor[46,500]
                    f_user_embed = self.embedding.user_embed(user_id)  # 一维[embed_size]
                    final_att = self.cat_attention(f_user_embed, h, h)
                else:
                    h = torch.cat((long_term_catted, short_term_state))  # concat long and short 二维tensor[46,500]
                    f_user_embed = self.embedding.user_embed(user_id)  # 一维[embed_size]
                    final_att = self.final_attention(f_user_embed, h, h)
                output = self.out_linear(final_att)

            else:  # 不区分偏好重要度，直接拼接
                final_att = long_term_prefer + short_term_prefer
                output = self.out_linear(final_att)  # 一维tensor[POI个数]

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)
        poi_loss = self.loss_func(pred, label)
        if settings.enable_CL:
            loss = poi_loss + settings.CL_weight * CL_loss
            self.track_parameters['poi_loss'] = poi_loss.item()
            self.track_parameters['CL_weight'] = settings.CL_weight
            self.track_parameters['CL_loss'] = CL_loss.item()
        else:
            loss = poi_loss

        return loss, output

    def predict(self, sample):
        test_loss, pred_raw = self.forward(sample, is_train=False)
        ranking = torch.sort(pred_raw, descending=True)[1]

        if settings.enable_drop:
            target = sample[-1][0, -1 - settings.drop_steps]
        else:
            target = sample[-1][0, -1]

        return ranking, target, test_loss

    def print_parameters(self, epoch):
        print(f'\n{settings.output_file_name} parameters epoch {epoch}:')
        # 将 self.track_parameters 中的所有参数打印到一行
        for key, value in self.track_parameters.items():
            print(f'{key}: {value}', end='\t')
        print('')


if __name__ == "__main__":
    pass
