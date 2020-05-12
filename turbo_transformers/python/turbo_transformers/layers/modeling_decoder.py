# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from typing import Union, Optional, Sequence
import torch
from .return_type import convert_returns_as_type, ReturnType

from .utils import try_convert, convert2tt_tensor, create_empty_if_none, AnyTensor

from onmt.modules.multi_headed_attn import MultiHeadedAttention as OnmtMultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward as OnmtPositionwiseFeedForward
from onmt.decoders.transformer import TransformerDecoderLayer as OnmtTransformerDecoderLayer

from torch.nn import LayerNorm as TorchLayerNorm

import enum
import numpy as np

__all__ = [
    'MultiHeadedAttention', 'PositionwiseFeedForward',
    'TransformerDecoderLayer'
]


class MultiHeadedAttention(cxx.MultiHeadedAttention):
    def __call__(self,
                 key_tensor: AnyTensor,
                 value_tensor: AnyTensor,
                 query_tensor: AnyTensor,
                 mask: Optional[AnyTensor],
                 layer_cache: Optional[dict] = None,
                 attn_type: str = None,
                 pre_layernorm: bool = False,
                 post_add: bool = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        """ Implement a MultiHeadedAttention of OpenNMT-py
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py

        Attention: Now layer_cache only contains Nones
        For self-dot Attention elements in dict `layer_cache` are Nones.
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py#L339
        """
        if mask is None:
            raise "mask of MultiHeadedAttention shall not be None"
        key_tensor = try_convert(key_tensor)
        value_tensor = try_convert(value_tensor)
        query_tensor = try_convert(query_tensor)
        mask = try_convert(mask)
        # TODO(jiaruifang) add layer_cache suuport in future
        if layer_cache is not None:
            for elem in layer_cache:
                assert layer_cache is None
        output = create_empty_if_none(output)

        super(MultiHeadedAttention,
              self).__call__(key_tensor, value_tensor, query_tensor, mask,
                             attn_type, output, pre_layernorm, post_add)

        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention):
        params = {k: v for k, v in multi_headed_attn.named_parameters()}
        # linear_keys.weight
        # linear_keys.bias
        # linear_values.weight
        # linear_values.bias
        # linear_query.weight
        # linear_query.bias
        # final_linear.weight
        # final_linear.bias
        if multi_headed_attn.max_relative_positions != 0:
            raise "multi_headed_attn's max_relative_positions should be 0!"

        # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
        qkv_weight = torch.clone(
            torch.t(
                torch.cat((params['linear_query.weight'],
                           params['linear_keys.weight'],
                           params['linear_values.weight']), 0)))
        qkv_bias = torch.cat(
            (params['linear_query.bias'], params['linear_keys.bias'],
             params['linear_values.bias']), 0)
        with torch.no_grad():
            att = MultiHeadedAttention(
                convert2tt_tensor(
                    torch.clone(torch.t(params['linear_keys.weight']))),
                convert2tt_tensor(params['linear_keys.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(params['linear_values.weight']))),
                convert2tt_tensor(params['linear_values.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(params['linear_query.weight']))),
                convert2tt_tensor(params['linear_query.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(params['final_linear.weight']))),
                convert2tt_tensor(params['final_linear.bias']),
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                multi_headed_attn.head_count)
            return att

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
                  layer_norm: TorchLayerNorm):
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        ln_params = {k: v for k, v in layer_norm.named_parameters()}

        qkv_weight = torch.clone(
            torch.t(
                torch.cat((attn_params['linear_query.weight'],
                           attn_params['linear_keys.weight'],
                           attn_params['linear_values.weight']), 0)))
        qkv_bias = torch.cat(
            (attn_params['linear_query.bias'], attn_params['linear_keys.bias'],
             attn_params['linear_values.bias']), 0)
        with torch.no_grad():
            att = MultiHeadedAttention(
                convert2tt_tensor(
                    torch.clone(torch.t(attn_params['linear_keys.weight']))),
                convert2tt_tensor(attn_params['linear_keys.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(attn_params['linear_values.weight']))),
                convert2tt_tensor(attn_params['linear_values.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(attn_params['linear_query.weight']))),
                convert2tt_tensor(attn_params['linear_query.bias']),
                convert2tt_tensor(
                    torch.clone(torch.t(attn_params['final_linear.weight']))),
                convert2tt_tensor(attn_params['final_linear.bias']),
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(ln_params['weight']),
                convert2tt_tensor(ln_params['bias']),
                multi_headed_attn.head_count)
            return att


class PositionwiseFeedForward(cxx.PositionwiseFeedForward):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(PositionwiseFeedForward, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_onmt(position_wise_ffn: OnmtPositionwiseFeedForward):
        params = {k: v for k, v in position_wise_ffn.named_parameters()}
        # w_1.weight
        # w_1.bias
        # w_2.weight
        # w_2.bias
        # layer_norm.weight
        # layer_norm.bias

        with torch.no_grad():
            ffn = PositionwiseFeedForward(
                convert2tt_tensor(torch.clone(torch.t(params['w_1.weight']))),
                convert2tt_tensor(params['w_1.bias']),
                convert2tt_tensor(torch.clone(torch.t(params['w_2.weight']))),
                convert2tt_tensor(params['w_2.bias']),
                convert2tt_tensor(params['layer_norm.weight']),
                convert2tt_tensor(params['layer_norm.bias']))
            return ffn


class TransformerDecoderLayer:
    def __init__(self, self_attn: MultiHeadedAttention,
                 context_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward):
        """ Implement class TransformerDecoderLayer(nn.Module):
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py
        self_attn_type of MultiHeadedAttention should always scaled-dot
        """
        self.self_attn = self_attn
        if not isinstance(self_attn, MultiHeadedAttention):
            raise "self_attn should be of type MultiHeadedAttention"
        self.context_attn = context_attn
        if not isinstance(context_attn, MultiHeadedAttention):
            raise "context_attn should be of type MultiHeadedAttention"
        self.feed_forward = feed_forward

    def __call__(self,
                 input_tensor: torch.Tensor,
                 memory_bank: torch.Tensor,
                 src_pad_mask: torch.Tensor,
                 tgt_pad_mask: torch.Tensor,
                 layer_cache: Optional[dict] = None,
                 step: Optional[int] = None,
                 future: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        """ Implement _forward method of class TransformerDecoderLayer
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py
        Because we now do not need context aligment, so we do not provide a forward method
        Args:
            input_tensor (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (FloatTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (FloatTensor): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``
        """

        # dec_mask = None
        dec_mask = torch.zeros(
            (input_tensor.size(0), 1, src_pad_mask.size(-1)),
            device=tgt_pad_mask.device,
            dtype=torch.float32)

        input_tensor = try_convert(input_tensor)
        memory_bank = try_convert(memory_bank)
        src_pad_mask = try_convert(src_pad_mask)

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones([tgt_len, tgt_len],
                                         device=tgt_pad_mask.device,
                                         dtype=torch.float32)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                # try:
                #     future_mask = future_mask.bool()
                # except AttributeError:
                #     pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0).float()
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        dec_mask = try_convert(dec_mask)

        query = self.self_attn(input_tensor,
                               input_tensor,
                               input_tensor,
                               mask=dec_mask,
                               layer_cache=layer_cache,
                               attn_type="self",
                               pre_layernorm=True,
                               post_add=True,
                               return_type=ReturnType.turbo_transformers)

        mid = self.context_attn(memory_bank,
                                memory_bank,
                                query,
                                mask=src_pad_mask,
                                layer_cache=layer_cache,
                                attn_type="context",
                                pre_layernorm=True,
                                post_add=True,
                                return_type=ReturnType.turbo_transformers)

        output = self.feed_forward(mid, return_type=return_type)
        return output, None
        # return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_onmt(transformer_decoder_layer: OnmtTransformerDecoderLayer):
        params = {
            k: v
            for k, v in transformer_decoder_layer.named_parameters()
        }
        # for k, v in transformer_decoder_layer.named_parameters():
        #     print(k, v.size())

        # 12: self_attn.linear_keys.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_keys.bias torch.Size([1024])
        # 12: self_attn.linear_values.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_values.bias torch.Size([1024])
        # 12: self_attn.linear_query.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_query.bias torch.Size([1024])
        # 12: self_attn.final_linear.weight torch.Size([1024, 1024])
        # 12: self_attn.final_linear.bias torch.Size([1024])
        # 12: context_attn.linear_keys.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_keys.bias torch.Size([1024])
        # 12: context_attn.linear_values.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_values.bias torch.Size([1024])
        # 12: context_attn.linear_query.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_query.bias torch.Size([1024])
        # 12: context_attn.final_linear.weight torch.Size([1024, 1024])
        # 12: context_attn.final_linear.bias torch.Size([1024])
        # 12: feed_forward.w_1.weight torch.Size([1, 1024])
        # 12: feed_forward.w_1.bias torch.Size([1])
        # 12: feed_forward.w_2.weight torch.Size([1024, 1])
        # 12: feed_forward.w_2.bias torch.Size([1024])
        # 12: feed_forward.layer_norm.weight torch.Size([1024])
        # 12: feed_forward.layer_norm.bias torch.Size([1024])
        # 12: layer_norm_1.weight torch.Size([1024])
        # 12: layer_norm_1.bias torch.Size([1024])
        # 12: layer_norm_2.weight torch.Size([1024])
        # 12: layer_norm_2.bias torch.Size([1024])
        # 12: w_1.weight torch.Size([1, 1024])
        # 12: w_1.bias torch.Size([1])
        # 12: w_2.weight torch.Size([1024, 1])
        # 12: w_2.bias torch.Size([1024])
        # 12: layer_norm.weight torch.Size([1024])
        # 12: layer_norm.bias torch.Size([1024])

        self_attn = MultiHeadedAttention.from_onmt(
            transformer_decoder_layer.self_attn,
            transformer_decoder_layer.layer_norm_1)
        context_attn = MultiHeadedAttention.from_onmt(
            transformer_decoder_layer.context_attn,
            transformer_decoder_layer.layer_norm_2)
        feed_forward = PositionwiseFeedForward.from_onmt(
            transformer_decoder_layer.feed_forward)

        return TransformerDecoderLayer(self_attn, context_attn, feed_forward)
