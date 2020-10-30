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
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor

from transformers.modeling_distilbert import DistilBertConfig
from transformers.modeling_distilbert import MultiHeadSelfAttention as TorchDistilMultiHeadSelfAttention

from torch import nn
import enum
import numpy as np

__all__ = ['DistillBertAttention']


class DistillBertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 is_trans_weight: Optional[cxx.Tensor] = False):
        assert (head_mask is None)
        # attention mask is different from BERT
        # attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0  #-float("inf")
        #.view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)

        input_tensor = try_convert(input_tensor)
        attention_mask = try_convert(create_empty_if_none(attention_mask))
        context_layer = cxx.Tensor.create_empty()
        attn_probs = cxx.Tensor.create_empty()
        super(DistillBertAttention,
              self).__call__(input_tensor, attention_mask, context_layer,
                             attn_probs, is_trans_weight)
        outputs = (convert_returns_as_type(context_layer, return_type),
                   convert_returns_as_type(attn_probs, ReturnType.TORCH)
                   ) if output_attentions else (convert_returns_as_type(
                       context_layer, return_type), )
        return outputs

    @staticmethod
    def from_torch(attention: TorchDistilMultiHeadSelfAttention,
                   layernorm: nn.LayerNorm):
        params = {k: v for k, v in attention.named_parameters()}
        layernorm_params = {k: v for k, v in layernorm.named_parameters()}
        # print(layernorm_params)
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['q_lin.weight'], params['k_lin.weight'],
                               params['v_lin.weight']),
                              0).contiguous()).contiguous())
            qkv_bias = torch.cat((params['q_lin.bias'], params['k_lin.bias'],
                                  params['v_lin.bias']), 0).contiguous()

            output_weight = torch.clone(
                torch.t(params['out_lin.weight']).contiguous())
            att = DistillBertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['out_lin.bias']),
                convert2tt_tensor(layernorm_params['weight']),
                convert2tt_tensor(layernorm_params['bias']), attention.n_heads)

            return att
