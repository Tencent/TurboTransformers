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
"""PyTorch ALBERT model. """
try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
import torch.utils.dlpack as dlpack
import numpy as np
from typing import Union, Optional, Sequence
from .return_type import convert_returns_as_type, ReturnType
from transformers.modeling_albert import AlbertEmbeddings as TorchAlbertEmbeddings
from transformers.modeling_albert import AlbertTransformer as TorchAlbertTransformer
from transformers.modeling_albert import AlbertAttention as TorchAlbertAttention
from transformers.modeling_albert import AlbertLayer as TorchAlbertLayer
from transformers.modeling_albert import AlbertLayerGroup as TorchAlbertLayerGroup
from transformers.modeling_albert import AlbertModel as TorchAlbertModel
from transformers.modeling_albert import AlbertConfig
import torch
from .utils import get_head_mask
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor, get_head_mask, get_extended_attention_mask

from torch import nn
import enum

__all__ = [
    "AlbertEmbeddings", "AlbertAttention", "AlbertLayerGroup", "AlbertLayer",
    "AlbertTransformer", "AlbertModel"
]


def _to_param_dict_naive(torch_module: torch.nn.Module):
    return {k: v for k, v in torch_module.named_parameters()}


# The implement of AlbertEmbeddings is totally equivalent with AlbertEmbedding, So nothing new to do
class AlbertEmbeddings(cxx.BERTEmbedding):
    def __call__(self,
                 input_ids: AnyTensor,
                 position_ids: AnyTensor,
                 token_type_ids: AnyTensor,
                 inputs_embeds: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        if inputs_embeds is not None:
            raise ("inputs_embeds must be None")
        input_ids = try_convert(input_ids)
        position_ids = try_convert(position_ids)
        token_type_ids = try_convert(token_type_ids)
        output = create_empty_if_none(output)
        super(AlbertEmbeddings, self).__call__(input_ids, position_ids,
                                               token_type_ids, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(albert_embedding: TorchAlbertEmbeddings
                   ) -> 'AlbertEmbeddings':
        params = to_param_dict_convert_tt(albert_embedding)
        return AlbertEmbeddings(params['word_embeddings.weight'],
                                params['position_embeddings.weight'],
                                params['token_type_embeddings.weight'],
                                params['LayerNorm.weight'],
                                params['LayerNorm.bias'])


class AlbertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: AnyTensor,
                 output_attentions: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        attention_mask = try_convert(attention_mask)
        output = create_empty_if_none(output)
        attn_probs = cxx.Tensor.create_empty()
        super(AlbertAttention, self).__call__(input_tensor, attention_mask,
                                              output, attn_probs, False)
        return (convert_returns_as_type(output, return_type),
                convert_returns_as_type(
                    attn_probs, return_type)) if output_attentions else (
                        convert_returns_as_type(output, return_type), )

    @staticmethod
    def from_torch(attention: TorchAlbertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['query.weight'], params['key.weight'],
                               params['value.weight']), 0)).contiguous())
            qkv_bias = torch.cat((params['query.bias'], params['key.bias'],
                                  params['value.bias']), 0)

            output_weight = torch.clone(
                torch.t(params['dense.weight']).contiguous())
            att = AlbertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['dense.bias']),
                convert2tt_tensor(params['LayerNorm.weight']),
                convert2tt_tensor(params['LayerNorm.bias']),
                attention.num_attention_heads)
            return att


class AlbertLayer(cxx.AlbertLayer):
    def __init__(self, attention: AlbertAttention, ffn, ffn_bias, ffn_output,
                 ffn_ouput_bias, fl, fl_bias):
        self.attention = attention
        super(AlbertLayer, self).__init__(ffn, ffn_bias, ffn_output,
                                          ffn_ouput_bias, fl, fl_bias)

    def __call__(
            self,
            input_tensor: AnyTensor,
            attention_mask: AnyTensor,
            head_mask: AnyTensor = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[
                bool] = False,  # a useless parameter in transformers v3.0.2
            return_type: Optional[ReturnType] = ReturnType.TORCH,
            attention_output: Optional[AnyTensor] = None):
        #TODO(jiaruifang) soldom see users use head_mask, so I am too lazy to add a head_mask.
        if head_mask is not None:
            raise ("head mask must be None")
        attention_output = create_empty_if_none(attention_output)
        attention_output = self.attention(
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
            return_type=ReturnType.turbo_transformers,
            output=attention_output)
        attention_output = try_convert(attention_output)
        hidden_output = create_empty_if_none(None)
        hidden_states = create_empty_if_none(None)
        super(AlbertLayer, self).__call__(attention_output[0], hidden_output,
                                          hidden_states)

        #TODO(jiaruifang) return_type conversion is conducted on the 1st return value, the other return values are TORCH tensors.
        return (convert_returns_as_type(hidden_states, return_type),
                convert_returns_as_type(attention_output[1], ReturnType.TORCH)
                ) if output_attentions else (convert_returns_as_type(
                    hidden_states, return_type), )

    @staticmethod
    def from_torch(torch_albert_layer: TorchAlbertLayer):
        ffn_params = _to_param_dict_naive(torch_albert_layer.ffn)
        ffn_output_params = _to_param_dict_naive(torch_albert_layer.ffn_output)
        full_layer_norm_params = _to_param_dict_naive(
            torch_albert_layer.full_layer_layer_norm)

        ffn_weight = torch.clone(torch.t(ffn_params["weight"]).contiguous())
        ffn_output_weight = torch.clone(
            torch.t(ffn_output_params["weight"]).contiguous())
        return AlbertLayer(
            AlbertAttention.from_torch(torch_albert_layer.attention),
            convert2tt_tensor(ffn_weight),
            convert2tt_tensor(ffn_params['bias']),
            convert2tt_tensor(ffn_output_weight),
            convert2tt_tensor(ffn_output_params['bias']),
            convert2tt_tensor(full_layer_norm_params['weight']),
            convert2tt_tensor(full_layer_norm_params['bias']))

    @staticmethod
    def from_npz(file_name: str, layer_num: int):
        raise ("Not Implemented")


class AlbertLayerGroup:
    def __init__(self, layers: Sequence[AlbertLayer]):
        self.albert_layers = layers

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor = None,
                 head_mask: AnyTensor = None,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 return_type: Optional[ReturnType] = None,
                 outputs: Optional[cxx.Tensor] = None):
        outputs = create_empty_if_none(outputs)
        layer_hidden_states = ()
        layer_attentions = ()
        hidden_states = try_convert(hidden_states)

        for layer_index, albert_layer in enumerate(self.albert_layers):
            # TODO(jiaruifang) head_mask must be None
            assert (head_mask[layer_index] is None)
            layer_output = albert_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                head_mask=head_mask[layer_index],
                return_type=ReturnType.turbo_transformers)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1], )

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )

        outputs = (convert_returns_as_type(hidden_states, return_type), )
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states, )
        if output_attentions:
            outputs = outputs + (layer_attentions, )
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)

    @staticmethod
    def from_torch(torch_layer_group: TorchAlbertLayerGroup):
        albertlayergroup = [
            AlbertLayer.from_torch(layer)
            for layer in torch_layer_group.albert_layers
        ]
        return AlbertLayerGroup(albertlayergroup)


# TODO(jiaruifang) Add Albert Transformers
class AlbertTransformer:
    """
    huggingface/transformer v3.0
    https://github.com/huggingface/transformers/blob/58cca47c16149e43d1b516623d59e3c5d97f695e/src/transformers/modeling_albert.py#L316
    """
    def __init__(self, embedding_hidden_mapping_in: nn.Linear,
                 albert_layer_groups: AlbertLayerGroup, config: AlbertConfig):
        self.config = config
        self.embedding_hidden_mapping_in = embedding_hidden_mapping_in
        self.albert_layer_groups = albert_layer_groups

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 head_mask: AnyTensor = None,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 return_type: Optional[ReturnType] = ReturnType.TORCH,
                 output: Optional[cxx.Tensor] = None):
        output = create_empty_if_none(output)
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        all_attentions = ()
        if output_hidden_states:
            all_hidden_states = (hidden_states, )
        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers /
                                   self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers /
                                 self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask=head_mask[group_idx *
                                    layers_per_group:(group_idx + 1) *
                                    layers_per_group],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_type=ReturnType.turbo_transformers)
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )

        outputs = (convert_returns_as_type(hidden_states, return_type), )
        if output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs

    @staticmethod
    def from_torch(torch_transformer_model: TorchAlbertTransformer):
        albert_layer_groups = [
            AlbertLayerGroup.from_torch(albert_layer_group) for
            albert_layer_group in torch_transformer_model.albert_layer_groups
        ]
        return AlbertTransformer(
            torch_transformer_model.embedding_hidden_mapping_in,
            albert_layer_groups, torch_transformer_model.config)


class AlbertModel:
    """
    huggingface/transformer v3.0
    https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_albert.py#L442
    """
    def __init__(self, embeddings: TorchAlbertEmbeddings,
                 encoder: AlbertTransformer, pooler: nn.Linear,
                 config: AlbertConfig):
        self.config = config
        self.embeddings = embeddings
        self.encoder = encoder
        self.pooler = pooler
        self.pooler_activation = nn.Tanh()

    def __call__(self,
                 input_ids=None,
                 attention_mask: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask=None,
                 inputs_embeds=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(
            self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:]  # add hidden_states and attentions if they are here
        return outputs

    @staticmethod
    def from_torch(torch_model: TorchAlbertModel):
        return AlbertModel(
            # AlbertEmbeddings.from_torch(torch_model.embeddings),
            torch_model.embeddings,
            AlbertTransformer.from_torch(torch_model.encoder),
            torch_model.pooler,
            torch_model.config)
