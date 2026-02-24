from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import linear, pad, softmax, dropout
from torch.nn.modules.linear import Linear
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import warnings

device = torch.device("cuda:0")
class MaskedTensorAttention(nn.Module):
    """
    Args:
        d_model: total dimension of the model.
        num_head: parallel attention heads.
        mode: choose from 'spatial' and 'temporal'
        dropout: a Dropout layer on attn_output_weights. Default: 0.1.
    """
    def __init__(self,
                 d_model: int,
                 num_head: int,
                 mode: str = 'temporal',
                 dropout: float = 0.1):
        super().__init__()
        assert mode in ['spatial', 'temporal']
        self.mode = mode
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, num_head, dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, obj_len, d_model = query.shape
        attn_dim = 2 if self.mode == 'spatial' else 1
        target_len = query.shape[attn_dim]
        source_len = key.shape[attn_dim]
        reserve_size = [query.shape[idx] for idx in range(len(query.shape) - 1) if idx != attn_dim]

        if key_padding_mask is not None:
            kp_mask = ~key_padding_mask.transpose(1, 2) if attn_dim == 1 else ~key_padding_mask
            kp_mask = kp_mask.reshape(-1, source_len)
        else:
            key_padding_mask = torch.ones(batch_size, seq_len, obj_len, 1).bool().to(device)
            kp_mask = ~key_padding_mask.transpose(1, 2) if attn_dim == 1 else ~key_padding_mask
            kp_mask = kp_mask.reshape(-1, source_len)

        a_mask = ~attn_mask if attn_mask is not None else None

        if a_mask is not None and attn_mask.shape[1] > 2:
            no_zero_mask = ~(torch.bmm((~kp_mask).unsqueeze(2).float(), (~kp_mask).unsqueeze(1).float()).bool())
            no_zero_mask += a_mask.unsqueeze(0)
            no_zero_mask = (no_zero_mask.reshape(kp_mask.size(0), -1).sum(dim=1)) < target_len * source_len
        else:
            no_zero_mask = kp_mask.sum(dim=1) < source_len

        q = query if attn_dim == 2 else query.permute(0, 2, 1, 3)
        k = key if attn_dim == 2 else key.permute(0, 2, 1, 3)
        v = value if attn_dim == 2 else value.permute(0, 2, 1, 3)

        q = q.reshape(-1, target_len, self.d_model)[no_zero_mask].permute(1, 0, 2)
        k = k.reshape(-1, source_len, self.d_model)[no_zero_mask].permute(1, 0, 2)
        v = v.reshape(-1, source_len, self.d_model)[no_zero_mask].permute(1, 0, 2)

        no_zero_output, no_zero_attn = self.attn(q, k, v, key_padding_mask=kp_mask[no_zero_mask], attn_mask=a_mask)

        output = no_zero_output.new_zeros((kp_mask.size(0), target_len, d_model))
        attn = no_zero_output.new_zeros((kp_mask.size(0), target_len, source_len))
        output[no_zero_mask] += no_zero_output.permute(1, 0, 2)
        output = output.permute(1, 0, 2)
        attn[no_zero_mask] += no_zero_attn

        if attn_dim == 1:
            output = output.reshape(batch_size, obj_len, seq_len, d_model).transpose(1, 2)
        else:
            output = output.reshape(batch_size, seq_len, obj_len, d_model)

        return output, attn.reshape(reserve_size + list(attn.size()[-2:]))


class CausalConvResidual(nn.Module):
    def __init__(self, d_model: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1,
                 norm_befor: bool = False):
        super().__init__()
        padding = kernel_size // 2
        self.conv_layers = nn.ModuleList()
        self.dilated_conv_layers = nn.ModuleList()
        for _ in range(2):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=padding, dilation=1),
                nn.ReLU(inplace=True)))
        for _ in range(2):
            self.dilated_conv_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation),
                nn.ReLU(inplace=True)))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_befor = norm_befor

    def forward(self, x: Tensor, masks: Optional[Tensor] = None):
        batch_size, seq_len, obj_len, d_model = x.shape

        no_zero_mask = masks.permute(0, 2, 3, 1).reshape(batch_size * obj_len, 1, seq_len).float() \
            if masks is not None else x.new_ones(batch_size * obj_len, 1, seq_len).float()

        x2 = self.norm(x) if self.norm_befor else x

        x2 = x2.permute(0, 2, 3, 1).reshape(batch_size * obj_len, d_model, seq_len)

        for idx in range(len(self.conv_layers)):
            x2 = self.conv_layers[idx](x2) * no_zero_mask
            x2 = self.dilated_conv_layers[idx](x2) * no_zero_mask
        x2 = x2.reshape(-1, obj_len, d_model, seq_len).permute(0, 3, 1, 2)

        x = x + self.dropout(x2)

        return self.norm(x) if not self.norm_befor else x


class TensorAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_head: int,
                 mode: str = 'temporal',
                 dropout: float = 0.1,
                 norm_before: bool = False):
        super().__init__()
        self.mode = mode
        self.mta = MaskedTensorAttention(d_model, num_head, mode, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_before = norm_before

    def forward(self,
                x: Union[Tensor, Tuple[Tensor, Tensor]],
                key_padding_mask: Optional[Tensor] = None,
                pos_embed: Union[Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]] = None,
                attn_mask: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor]:
        if isinstance(x, Tensor):
            x2 = self.norm(x) if self.norm_before else x
            q = k = self._with_pos_embed(x2, pos_embed)
            x2, attn = self.mta(q, k, x2, key_padding_mask, attn_mask)
            x = x + self.dropout(x2)
            x = self.norm(x) if not self.norm_before else x
            return x, attn

        elif isinstance(x, Tuple) and len(x) == 2 and isinstance(pos_embed, Tuple) and len(pos_embed) == 2:
            dec, enc = x
            query_pos_embed, enc_pos_embed = pos_embed
            dec2 = self.norm(dec) if self.norm_before else dec
            dec2, dec_enc_attn = self.mta(self._with_pos_embed(dec2, query_pos_embed),
                                          self._with_pos_embed(enc, enc_pos_embed),
                                          enc,
                                          key_padding_mask,
                                          attn_mask)
            dec = dec + self.dropout(dec2)
            dec = self.norm(dec) if not self.norm_before else dec
            return dec, dec_enc_attn

        else:
            raise NotImplementedError

    @staticmethod
    def _with_pos_embed(tensor: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        return tensor if pos is None else tensor + pos

class PoswiseFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hid: int,
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 norm_before: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hid)
        self.act = getattr(F, activation)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hid, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.norm_before = norm_before
    
    def forward(self, x: Tensor) -> Tensor:
        x2 = self.norm(x) if self.norm_before else x
        x2 = self.fc2(self.dropout1(self.act(self.fc1(x2))))
        x = x + self.dropout2(x2)
        return self.norm(x) if not self.norm_before else x

class EncoderLayer(nn.Module):
    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = False,
                 with_global_conv: bool = True):
        super().__init__()
        self.cross_spa_tab = TensorAttentionBlock(d_model, num_head, mode='spatial', dropout=dropout,
                                                  norm_before=norm_before)
        self.enc_spa_tab = TensorAttentionBlock(d_model, num_head, mode='spatial', dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv1 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.enc_tem_tab = TensorAttentionBlock(d_model, num_head, mode='temporal', dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv2 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.ff = PoswiseFeedForward(d_model, d_hid, activation, dropout, norm_before)

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                temporal_pos_embed: Optional[Tensor] = None,
                map_embed: Optional[Tensor] = None):
        x, spa_attn = self.cross_spa_tab((x, map_embed.repeat(1, x.shape[1], 1, 1)),
                                         key_padding_mask=None,
                                         pos_embed=(None, None))
        enc_spa_output, enc_spa_attn = self.enc_spa_tab(x,
                                                        key_padding_mask=key_padding_mask)

        enc_spa_output = self.causal_conv1(enc_spa_output, key_padding_mask) \
            if self.causal_conv1 is not None else enc_spa_output

        enc_tem_output, enc_tem_attn = self.enc_tem_tab(enc_spa_output,
                                                        key_padding_mask=key_padding_mask,
                                                        pos_embed=temporal_pos_embed)

        enc_tem_output = self.causal_conv2(enc_tem_output, key_padding_mask) \
            if self.causal_conv2 is not None else enc_tem_output

        return self.ff(enc_tem_output), enc_spa_attn, enc_tem_attn


class Encoder(nn.Module):
    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 num_stack: int = 2,
                 activation: str = "relu",
                 dropout: float = 0.3,
                 norm_before: bool = True):
        super().__init__()
        self.encoder_stack = nn.ModuleList([
            EncoderLayer(num_head, d_model, d_hid, activation=activation,
                         dropout=dropout, norm_before=norm_before) for _ in range(num_stack)
        ])
        self._reset_parameters()

    def forward(self,
                x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                temporal_pos_embed: Optional[Tensor] = None,
                map_embed: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        enc_output = x
        enc_spa_attn_list = []
        enc_tem_attn_list = []
        for idx, layer in enumerate(self.encoder_stack):
            enc_output, enc_spa_attn, enc_tem_attn = layer(enc_output, key_padding_mask, temporal_pos_embed, map_embed)
            enc_spa_attn_list.append(enc_spa_attn)
            enc_tem_attn_list.append(enc_tem_attn)

        return enc_output, tuple(enc_spa_attn_list), tuple(enc_tem_attn_list)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class DecoderLayer(nn.Module):
    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 norm_before: bool = False,
                 with_global_conv: bool = True):
        super().__init__()
        self.cross_spa_tab = TensorAttentionBlock(d_model, num_head, mode='spatial', dropout=dropout,
                                                  norm_before=norm_before)
        self.dec_slf_tab = TensorAttentionBlock(d_model, num_head, mode='spatial', dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv1 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.dec_enc_tab = TensorAttentionBlock(d_model, num_head, mode='temporal', dropout=dropout,
                                                norm_before=norm_before)
        self.causal_conv2 = CausalConvResidual(d_model, kernel_size=3, dilation=3,
                                               dropout=dropout, norm_befor=norm_before) if with_global_conv else None
        self.ff = PoswiseFeedForward(d_model, d_hid, activation, dropout, norm_before)

    def forward(self,
                dec: Tensor,
                enc: Tensor,
                enc_key_padding_mask: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None,
                temporal_pos_embed: Optional[Tensor] = None,
                map_embed: Optional[Tensor] = None):
        dec, spa_attn = self.cross_spa_tab((dec, map_embed.repeat(1, dec.shape[1], 1, 1)),
                                            key_padding_mask=None,
                                            pos_embed=(None, None))

        dec_key_padding_mask = enc_key_padding_mask

        dec_spa_output, dec_slf_attn = self.dec_slf_tab(dec,
                                                        key_padding_mask=dec_key_padding_mask)

        dec_spa_output = self.causal_conv1(dec_spa_output, dec_key_padding_mask) \
            if self.causal_conv1 is not None else dec_spa_output

        dec_tem_output, dec_enc_attn = self.dec_enc_tab((dec_spa_output, enc),
                                                        key_padding_mask=enc_key_padding_mask,
                                                        pos_embed=(query_pos_embed, temporal_pos_embed))

        dec_tem_output = self.causal_conv2(dec_tem_output, dec_key_padding_mask) \
            if self.causal_conv2 is not None else dec_tem_output

        return self.ff(dec_tem_output), dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self,
                 num_head: int,
                 d_model: int,
                 d_hid: int,
                 num_stack: int = 2,
                 activation: str = "relu",
                 dropout: float = 0.3,
                 norm_before: bool = True):
        super().__init__()
        self.decoder_stack = nn.ModuleList([
            DecoderLayer(num_head, d_model, d_hid, activation=activation,
                         dropout=dropout, norm_before=norm_before) for _ in range(num_stack)
        ])
        self._reset_parameters()

    def forward(self,
                dec: Tensor,
                enc: Tensor,
                enc_key_padding_mask: Optional[Tensor] = None,
                query_pos_embed: Optional[Tensor] = None,
                temporal_pos_embed: Optional[Tensor] = None,
                map_embed: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        dec_output = dec
        dec_slf_attn_list = []
        dec_enc_attn_list = []
        for idx, layer in enumerate(self.decoder_stack):
            dec_output, dec_slf_attn, dec_enc_attn = layer(dec_output, enc, enc_key_padding_mask, query_pos_embed, temporal_pos_embed, map_embed)
            dec_slf_attn_list.append(dec_slf_attn)
            dec_enc_attn_list.append(dec_enc_attn)
        return dec_output, tuple(dec_slf_attn_list), tuple(dec_enc_attn_list)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)