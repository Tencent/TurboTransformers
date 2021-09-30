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
from transformers.models.bert.modeling_bert import BertAttention as TorchBertAttention
from onmt.modules.position_ffn import PositionwiseFeedForward as OnmtPositionwiseFeedForward
from torch.nn import LayerNorm as TorchLayerNorm

__all__ = [
    'MultiHeadedAttention',
    'PositionwiseFeedForward',
]


class MultiHeadedAttention(cxx.MultiHeadedAttention):
    def __call__(self,
                 key_tensor: AnyTensor,
                 value_tensor: AnyTensor,
                 query_tensor: AnyTensor,
                 mask: Optional[AnyTensor] = None,
                 layer_cache: Optional[dict] = None,
                 attn_type: str = None,
                 pre_layernorm: bool = False,
                 post_layernorm: bool = False,
                 post_add_input: bool = False,
                 is_trans_weight: bool = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None,
                 attn: Optional[cxx.Tensor] = None):
        """ Implement a MultiHeadedAttention of OpenNMT-py
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py

        Attention: Now layer_cache only contains Nones
        For self-dot Attention elements in dict `layer_cache` are Nones.
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py#L339
        """
        key_tensor = try_convert(key_tensor)
        value_tensor = try_convert(value_tensor)
        query_tensor = try_convert(query_tensor)

        mask = try_convert(create_empty_if_none(mask))

        output = create_empty_if_none(output)
        attn = create_empty_if_none(attn)
        layer_cache_tmp = {}
        if layer_cache is not None:
            for k, v in layer_cache.items():
                if v is not None:
                    layer_cache_tmp[k] = try_convert(v)
                else:
                    layer_cache_tmp[k] = create_empty_if_none(v)

        super(MultiHeadedAttention,
              self).__call__(key_tensor, value_tensor, query_tensor, mask,
                             attn_type, output, attn, layer_cache_tmp,
                             pre_layernorm, post_layernorm, post_add_input,
                             is_trans_weight)

        if layer_cache is not None:
            for k, v in layer_cache_tmp.items():
                if "memory" in k and "context" in attn_type or "self" in k and "self" in attn_type:
                    layer_cache[k] = convert_returns_as_type(
                        v, ReturnType.TORCH)

        return convert_returns_as_type(output,
                                       return_type), convert_returns_as_type(
                                           attn, return_type)

    @staticmethod
    def pack_parameter(multi_headed_attn: OnmtMultiHeadedAttention,
                       is_trans_weight: Optional[bool] = False):
        # linear_keys.weight
        # linear_keys.bias
        # linear_values.weight
        # linear_values.bias
        # linear_query.weight
        # linear_query.bias
        # final_linear.weight
        # final_linear.bias
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        if multi_headed_attn.max_relative_positions != 0:
            raise "multi_headed_attn's max_relative_positions should be 0!"

        # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
        if is_trans_weight:
            qkv_weight = torch.cat((attn_params['linear_query.weight'],
                                    attn_params['linear_keys.weight'],
                                    attn_params['linear_values.weight']), 0)
            k_w = convert2tt_tensor(attn_params['linear_keys.weight'])
            v_w = convert2tt_tensor(attn_params['linear_values.weight'])
            q_w = convert2tt_tensor(attn_params['linear_query.weight'])
            f_w = convert2tt_tensor(attn_params['final_linear.weight'])
        else:
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((attn_params['linear_query.weight'],
                               attn_params['linear_keys.weight'],
                               attn_params['linear_values.weight']),
                              0).contiguous()).contiguous())
            k_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_keys.weight']).contiguous()))
            v_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_values.weight']).contiguous()))
            q_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_query.weight']).contiguous()))
            f_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['final_linear.weight']).contiguous()))

        qkv_bias = torch.cat(
            (attn_params['linear_query.bias'], attn_params['linear_keys.bias'],
             attn_params['linear_values.bias']), 0)
        return (k_w, convert2tt_tensor(attn_params['linear_keys.bias']), v_w,
                convert2tt_tensor(attn_params['linear_values.bias']), q_w,
                convert2tt_tensor(attn_params['linear_query.bias']), f_w,
                convert2tt_tensor(attn_params['final_linear.bias']),
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias))

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
                  is_trans_weight: bool = False):
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        if multi_headed_attn.max_relative_positions != 0:
            raise "multi_headed_attn's max_relative_positions should be 0!"

        with torch.no_grad():
            att = MultiHeadedAttention(
                *(MultiHeadedAttention.pack_parameter(attn_params,
                                                      is_trans_weight)),
                multi_headed_attn.head_count)
            return att

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
                  layer_norm: TorchLayerNorm,
                  is_trans_weight: bool = False):
        ln_params = {k: v for k, v in layer_norm.named_parameters()}
        with torch.no_grad():
            att = MultiHeadedAttention(
                *(MultiHeadedAttention.pack_parameter(multi_headed_attn,
                                                      is_trans_weight)),
                convert2tt_tensor(ln_params['weight']),
                convert2tt_tensor(ln_params['bias']),
                multi_headed_attn.head_count)
            return att

    @staticmethod
    def from_torch(attention: TorchBertAttention,
                   layer_norm: Optional[TorchLayerNorm] = None,
                   is_trans_weight: bool = False):
        """
        load an attn model from huggingface bert attention model.
        """
        ln_params = {}
        if layer_norm is not None:
            ln_params = {k: v for k, v in layer_norm.named_parameters()}
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            if is_trans_weight:
                # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
                qkv_weight = torch.cat(
                    (params['self.query.weight'], params['self.key.weight'],
                     params['self.value.weight']), 0)
                output_weight = params['output.dense.weight']
                k_w = params['self.key.weight']
                v_w = params['self.value.weight']
                q_w = params['self.query.weight']
            else:
                # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
                qkv_weight = torch.clone(
                    torch.t(
                        torch.cat((params['self.query.weight'],
                                   params['self.key.weight'],
                                   params['self.value.weight']),
                                  0).contiguous()).contiguous())
                output_weight = torch.clone(
                    torch.t(params['output.dense.weight']).contiguous())
                k_w = torch.clone(
                    torch.t(params['self.key.weight']).contiguous())
                v_w = torch.clone(
                    torch.t(params['self.value.weight']).contiguous())
                q_w = torch.clone(
                    torch.t(params['self.query.weight']).contiguous())

            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0)

            if layer_norm is not None:
                att = MultiHeadedAttention(
                    convert2tt_tensor(k_w),
                    convert2tt_tensor(params['self.key.bias']),
                    convert2tt_tensor(v_w),
                    convert2tt_tensor(params['self.value.bias']),
                    convert2tt_tensor(q_w),
                    convert2tt_tensor(params['self.query.bias']),
                    convert2tt_tensor(output_weight),
                    convert2tt_tensor(params['output.dense.bias']),
                    convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                    convert2tt_tensor(params['output.LayerNorm.weight']),
                    convert2tt_tensor(params['output.LayerNorm.bias']),
                    convert2tt_tensor(ln_params['weight']),
                    convert2tt_tensor(ln_params['bias']),
                    attention.self.num_attention_heads)
            else:
                att = MultiHeadedAttention(
                    convert2tt_tensor(k_w),
                    convert2tt_tensor(params['self.key.bias']),
                    convert2tt_tensor(v_w),
                    convert2tt_tensor(params['self.value.bias']),
                    convert2tt_tensor(q_w),
                    convert2tt_tensor(params['self.query.bias']),
                    convert2tt_tensor(output_weight),
                    convert2tt_tensor(params['output.dense.bias']),
                    convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                    convert2tt_tensor(params['output.LayerNorm.weight']),
                    convert2tt_tensor(params['output.LayerNorm.bias']),
                    attention.self.num_attention_heads)
            return att

    @staticmethod
    def from_npz(file_name: str, layer_num: int, num_attention_heads: int):
        f = np.load(file_name)
        return BertAttention(
            create_empty_if_none(None), create_empty_if_none(None),
            create_empty_if_none(None), create_empty_if_none(None),
            create_empty_if_none(None), create_empty_if_none(None),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.weight']),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.bias']),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.weight']),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.bias']),
            try_convert(f[
                f'encoder.layer.{layer_num}.attention.output.LayerNorm.weight']
                        ),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.bias']
            ), num_attention_heads)


class PositionwiseFeedForward(cxx.PositionwiseFeedForward):
    def __call__(
            self,
            input_tensor: AnyTensor,
            return_type: Optional[ReturnType] = None,
            is_trans_weight: Optional[bool] = True,  #Intel 61xx True is faster
            output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(PositionwiseFeedForward, self).__call__(input_tensor, output,
                                                      is_trans_weight)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_onmt(position_wise_ffn: OnmtPositionwiseFeedForward,
                  is_trans_weight: Optional[bool] = True):
        params = {k: v for k, v in position_wise_ffn.named_parameters()}
        # w_1.weight
        # w_1.bias
        # w_2.weight
        # w_2.bias
        # layer_norm.weight
        # layer_norm.bias

        # Note that torch's weights of linear layer is transposed
        if is_trans_weight:
            w_1 = convert2tt_tensor(params['w_1.weight'])
            w_2 = convert2tt_tensor(params['w_2.weight'])
        else:
            w_1 = convert2tt_tensor(
                torch.clone(torch.t(params['w_1.weight']).contiguous()))
            w_2 = convert2tt_tensor(
                torch.clone(torch.t(params['w_2.weight']).contiguous()))

        with torch.no_grad():
            ffn = PositionwiseFeedForward(
                w_1, convert2tt_tensor(params['w_1.bias']), w_2,
                convert2tt_tensor(params['w_2.bias']),
                convert2tt_tensor(params['layer_norm.weight']),
                convert2tt_tensor(params['layer_norm.bias']))
            return ffn
