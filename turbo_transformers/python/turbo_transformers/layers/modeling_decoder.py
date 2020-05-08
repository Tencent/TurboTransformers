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

import enum
import numpy as np

__all__ = ['MultiHeadedAttention', 'PositionwiseFeedForward']


class MultiHeadedAttention(cxx.MultiHeadedAttention):
    def __call__(self,
                 key_tensor: AnyTensor,
                 value_tensor: AnyTensor,
                 query_tensor: AnyTensor,
                 mask_tensor: Optional[AnyTensor] = None,
                 layer_cache: Optional[AnyTensor] = None,
                 attn_type: str = None,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        key_tensor = try_convert(key_tensor)
        value_tensor = try_convert(value_tensor)
        query_tensor = try_convert(query_tensor)
        mask_tensor = try_convert(mask_tensor)
        # assert layer_cache == None
        # assert attn_type == "context"
        output = create_empty_if_none(output)
        super(MultiHeadedAttention,
              self).__call__(key_tensor, value_tensor, query_tensor,
                             mask_tensor, attn_type, output)
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
        for k, v in position_wise_ffn.named_parameters():
            print(k, v.size())
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
