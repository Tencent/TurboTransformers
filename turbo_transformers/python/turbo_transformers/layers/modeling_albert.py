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

from transformers.modeling_bert import AlbertModel as TorchAlbertModel

from transformers.modeling_bert import BertPooler as TorchBertPooler
from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.modeling_bert import BertOutput as TorchBertOutput
from onmt.modules.multi_headed_attn import MultiHeadedAttention as OnmtMultiHeadedAttention
from transformers.modeling_bert import BertLayer as TorchBertLayer
from transformers.modeling_bert import BertEncoder as TorchBertEncoder

from transformers.modeling_albert import AlbertLayer as TorchAlbertLayer
from transformers.modeling_albert import AlbertLayerGroup as TorchAlbertLayerGroup
from transformers.modeling_albert import AlbertTransformer as TorchAlbertTransformer
from torch.nn import LayerNorm as TorchLayerNorm

from .modeling_bert import BertPooler
from .modeling_decoder import MultiHeadedAttention

import enum
import numpy as np

__all__ = ['AlbertModel']

# TODO implement a Cpp class to do FFNs operations. activation is gelu_new
# class AlbertFFNs(cxx.AlbertFFNs):
#     r"""
#     implement parts as follows
#     ```
#     self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
#     self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
#     self.activation = ACT2FN[config.hidden_act]
#     ```
#     https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_albert.py#L265
#     """
#     def __call__(self,
#                  hidden_states: AnyTensor,
#                  attention_mask: Optional[AnyTensor] = None,
#                  head_mask: Optional[AnyTensor] = None,
#                  return_type: Optional[ReturnType] = None,
#                  hidden_states: Optional[cxx.Tensor] = None,
#                  attention_output: Optional[cxx.Tensor] = None):
#         pass
#     @staticmethod
#     def from_torch(ffn : nn.Linear, ffn_output : nn.Linear):
#         params = {k: v for k, v in attention.named_parameters()}
#         with torch.no_grad():
#             pass


class AlbertLayer:
    r"""
    implement this
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_albert.py#L265
    """
    def __init__(self, attn: OnmtMultiHeadedAttention, ffns: AlbertFFNs):
        self.attn = attn
        self.ffns = ffns

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 attention_output: Optional[cxx.Tensor] = None,
                 intermediate_output: Optional[cxx.Tensor] = None,
                 output: Optional[cxx.Tensor] = None):
        pass

    @staticmethod
    def from_torch(albertlayer: TorchAlbertLayer):
        return BertLayer(
            OnmtMultiHeadedAttention.from_torch(
                albertlayer.attention, albertlayer.full_layer_layer_norm),
            AlbertFFNs.from_torch(albertlayer.ffn, albertlayer.ffn_output))


class AlbertLayerGroup:
    r""" implement this
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_albert.py#L286
    """
    def __init__(self, layers: Sequence[AlbertLayer]):
        self.layers = layers

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 attention_output: Optional[cxx.Tensor] = None,
                 intermediate_output: Optional[cxx.Tensor] = None,
                 output: Optional[cxx.Tensor] = None):
        pass

    @staticmethod
    def from_torch(albert_layer_group: AlbertLayerGroup):
        layers = [
            AlbertLayer.from_torch(layer)
            for layer in AlbertLayerGroup.albert_layers
        ]
        return AlbertLayerGroup(layers)


class AlbertTransformer:
    r"""
    class AlbertTransformer(nn.Module):
    """
    def __init__(self, embedding_hidden_mapping_in: nn.Linear,
                 albert_layer_groups: Sequence[AlbertTransformer]):
        self.embedding_hidden_mapping_in = embedding_hidden_mapping_in
        self.albert_layer_groups = albert_layer_groups

    def __call__(self,
                 input_ids: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask: Optional[dict] = None,
                 return_type: Optional[ReturnType] = None,
                 last_hidden_state: Optional[cxx.Tensor] = None,
                 pooler_output: Optional[cxx.Tensor] = None,
                 hidden_states: Optional[cxx.Tensor] = None,
                 attentions: Optional[Sequence] = None):
        pass

    @staticmethod
    def from_torch(transformer: TorchAlbertTransformer):
        layers = [
            AlbertLayer.from_torch(albert_layer_group)
            for albert_layer_group in AlbertLayerGroup.albert_layer_groups
        ]
        return AlbertLayerGroup(albert.embedding_hidden_mapping_in,
                                albert.encoder)


class AlbertModel:
    r""" Implement Albert
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_albert.py#L439
    """
    def __init__(self, embeddings: BertEmbeddings, encoder: AlbertTransformer,
                 pooler: BertPooler):
        self.embeddings = embeddings
        self.encoder = encoder
        self.prepare = cxx.PrepareBertMasks()

    def __call__(self,
                 input_ids: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask: Optional[dict] = None,
                 return_type: Optional[ReturnType] = None,
                 last_hidden_state: Optional[cxx.Tensor] = None,
                 pooler_output: Optional[cxx.Tensor] = None,
                 hidden_states: Optional[cxx.Tensor] = None,
                 attentions: Optional[Sequence] = None):
        pass

    @staticmethod
    def from_torch(albert: TorchAlbertModel):
        return AlbertModel(albert.embeddings, albert.encoder, albert.pooler)
