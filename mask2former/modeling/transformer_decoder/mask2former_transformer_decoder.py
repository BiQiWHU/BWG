# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

import numpy as np


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # hilo 0.5
        # self.self_attn_hi = nn.MultiheadAttention(d_model, nhead // 2, dropout=dropout)
        # self.self_attn_lo = nn.MultiheadAttention(d_model // 4, nhead // 2, dropout=dropout)
        # hilo 1 
        # self.self_attn_hi = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn_lo = nn.MultiheadAttention(d_model // 4, nhead, dropout=dropout)
        # hilo wavelet 1
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.wavepool_1 = WavePool(1)
        self.wavepool_2 = WavePool(1)
        self.wavepool_3 = WavePool(1)
        self.instance_norm = nn.InstanceNorm1d(256)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        # self.avgpool = nn.AvgPool2d(2, stride=2)
        # hilo 0.5
        # self.concentrate_linear = nn.Linear(d_model+d_model//4, d_model)
        # hilo 1
        # self.concentrate_linear = nn.Linear(d_model+d_model//4, d_model)
        # hilo wavelet 1
        # self.concentrate_linear = nn.Linear(d_model*3, d_model)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # def forward_post(self, tgt,
    #                  tgt_mask: Optional[Tensor] = None,
    #                  tgt_key_padding_mask: Optional[Tensor] = None,
    #                  query_pos: Optional[Tensor] = None):
    #     q = k = self.with_pos_embed(tgt, query_pos)
    #     tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout(tgt2)
    #     tgt = self.norm(tgt)

    #     return tgt

    # def forward_pre(self, tgt,
    #                 tgt_mask: Optional[Tensor] = None,
    #                 tgt_key_padding_mask: Optional[Tensor] = None,
    #                 query_pos: Optional[Tensor] = None):
    #     tgt2 = self.norm(tgt)
    #     q = k = self.with_pos_embed(tgt2, query_pos)
    #     tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout(tgt2)
        
    #     return tgt
    
    # ### HiLo Self-Attention
    # def forward_post(self, tgt,
    #                  tgt_mask: Optional[Tensor] = None,
    #                  tgt_key_padding_mask: Optional[Tensor] = None,
    #                  query_pos: Optional[Tensor] = None):
    #     # hight freq
    #     q_hi = k_hi = self.with_pos_embed(tgt, query_pos)
    #     v_hi = tgt
    #     tgt2_hi = self.self_attn_hi(q_hi, k_hi, value=v_hi, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     # low freq
    #     B, N, C = tgt.shape
    #     tgt_lo = self.avgpool(tgt.view((B, N, int(np.sqrt(C)), -1))).view((B, N, -1))
    #     # q_lo = self.lo_linear(tgt)
    #     q_lo = k_lo = self.with_pos_embed(tgt_lo, None)
    #     v_lo = tgt_lo
    #     tgt2_lo = self.self_attn_lo(q_lo, k_lo, value=v_lo, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt2 = torch.cat((tgt2_hi, tgt2_lo), -1)
    #     tgt2 = self.concentrate_linear(tgt2)
    #     tgt = tgt + self.dropout(tgt2)
    #     tgt = self.norm(tgt)

    #     return tgt

    # def forward_pre(self, tgt,
    #                 tgt_mask: Optional[Tensor] = None,
    #                 tgt_key_padding_mask: Optional[Tensor] = None,
    #                 query_pos: Optional[Tensor] = None):
    #     tgt2 = self.norm(tgt)
    #     # hight freq
    #     q_hi = k_hi = self.with_pos_embed(tgt, query_pos)
    #     v_hi = tgt
    #     tgt2_hi = self.self_attn_hi(q_hi, k_hi, value=v_hi, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     # low freq
    #     B, N, C = tgt.shape
    #     tgt_lo = self.avgpool(tgt.view((B, N, int(np.sqrt(C)), -1))).view((B, N, -1))
    #     # q_lo = self.lo_linear(tgt)
    #     q_lo = k_lo = self.with_pos_embed(tgt_lo, None)
    #     v_lo = tgt_lo
    #     tgt2_lo = self.self_attn_lo(q_lo, k_lo, value=v_lo, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt2 = torch.cat((tgt2_hi, tgt2_lo), -1)
    #     tgt2 = self.concentrate_linear(tgt2)
    #     tgt = tgt + self.dropout(tgt2)
        
    #     return tgt
    # ### HiLo Self-Attention
    
    ### HiLo Wavelet Self-Attention
    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        B, N, C = tgt.shape
        # tgt2 [100, 1, 256]
        tgt2_1 = self.self_attn_1(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_2 = self.self_attn_2(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_3 = self.self_attn_3(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        hw = int(np.sqrt(C))
        tgt2_1 = tgt2_1.reshape(B, N, hw, hw)
        tgt2_2 = tgt2_2.reshape(B, N, hw, hw)
        tgt2_3 = tgt2_3.reshape(B, N, hw, hw)
        ll_1, lh_1, hl_1, hh_1 = self.wavepool_1(tgt2_1)
        ll_2, lh_2, hl_2, hh_2 = self.wavepool_2(tgt2_2)
        ll_3, lh_3, hl_3, hh_3 = self.wavepool_3(tgt2_3)
        c = int(C / 4)
        ll_1, lh_1, hl_1, hh_1 = ll_1.reshape(B, N, c), lh_1.reshape(B, N, c), hl_1.reshape(B, N, c), hh_1.reshape(B, N, c)
        ll_2, lh_2, hl_2, hh_2 = ll_2.reshape(B, N, c), lh_2.reshape(B, N, c), hl_2.reshape(B, N, c), hh_2.reshape(B, N, c)
        ll_3, lh_3, hl_3, hh_3 = ll_3.reshape(B, N, c), lh_3.reshape(B, N, c), hl_3.reshape(B, N, c), hh_3.reshape(B, N, c)
        # out_2 = (ll_1 + ll_2 + ll_3) / 3
        # out_1 = (lh_1 + hl_1 + hh_1 + lh_2 + hl_2 + hh_2) / 6
        # out_3 = (lh_1 + hl_1 + hh_1 + lh_3 + hl_3 + hh_3) / 6
        out_1 = torch.cat((lh_1, hl_1, hh_1, (lh_2 + hl_2 + hh_2) / 3), -1)
        out_2 = torch.cat((ll_2, ll_2, ll_1, ll_3), -1)
        out_3 = torch.cat((lh_3, hl_3, hh_3, (lh_2 + hl_2 + hh_2) / 3), -1)
        out_1 = self.instance_norm(out_1)
        out_3 = self.instance_norm(out_3)
        tgt2 = (out_1 + out_2 + out_3) / 3
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        B, N, C = tgt.shape
        # tgt2 [100, 1, 256]
        tgt2_1 = self.self_attn_1(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_2 = self.self_attn_2(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt2_3 = self.self_attn_3(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        hw = int(np.sqrt(C))
        tgt2_1 = tgt2_1.reshape(B, N, hw, hw)
        tgt2_2 = tgt2_2.reshape(B, N, hw, hw)
        tgt2_3 = tgt2_3.reshape(B, N, hw, hw)
        ll_1, lh_1, hl_1, hh_1 = self.wavepool_1(tgt2_1)
        ll_2, lh_2, hl_2, hh_2 = self.wavepool_2(tgt2_2)
        ll_3, lh_3, hl_3, hh_3 = self.wavepool_3(tgt2_3)
        c = int(C / 4)
        ll_1, lh_1, hl_1, hh_1 = ll_1.reshape(B, N, c), lh_1.reshape(B, N, c), hl_1.reshape(B, N, c), hh_1.reshape(B, N, c)
        ll_2, lh_2, hl_2, hh_2 = ll_2.reshape(B, N, c), lh_2.reshape(B, N, c), hl_2.reshape(B, N, c), hh_2.reshape(B, N, c)
        ll_3, lh_3, hl_3, hh_3 = ll_3.reshape(B, N, c), lh_3.reshape(B, N, c), hl_3.reshape(B, N, c), hh_3.reshape(B, N, c)
        # out_2 = (ll_1 + ll_2 + ll_3) / 3
        # out_1 = (lh_1 + hl_1 + hh_1 + lh_2 + hl_2 + hh_2) / 6
        # out_3 = (lh_1 + hl_1 + hh_1 + lh_3 + hl_3 + hh_3) / 6
        out_1 = torch.cat((lh_1, hl_1, hh_1, (lh_2 + hl_2 + hh_2) / 3), -1)
        out_2 = torch.cat((ll_2, ll_2, ll_1, ll_3), -1)
        out_3 = torch.cat((lh_3, hl_3, hh_3, (lh_2 + hl_2 + hh_2) / 3), -1)
        out_1 = self.instance_norm(out_1)
        out_3 = self.instance_norm(out_3)
        tgt2 = (out_1 + out_2 + out_3) / 3
        tgt = tgt + self.dropout(tgt2)
        
        return tgt
    ### HiLo Wavelet Self-Attention

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
       
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
    
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
