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
"""PyTorch ROBERTA model. """
try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
import torch.utils.dlpack as dlpack
import numpy as np
from typing import Union, Optional, Sequence
from .return_type import convert_returns_as_type, ReturnType
import torch
from torch import nn
import enum
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor, get_head_mask, get_extended_attention_mask
from transformers.modeling_roberta import RobertaModel as TorchRobertaModel
from transformers.modeling_roberta import RobertaEmbeddings as TorchRobertaEmbeddings
from transformers.modeling_roberta import RobertaConfig
from transformers.modeling_bert import BertEncoder as TorchBertEncoder
from .modeling_bert import BertEncoder, SequencePool, BertPooler

__all__ = [" RobertaModel"]


# TODO move to utils
class PoolingType(enum.Enum):
    FIRST = "First"
    LAST = "Last"
    MEAN = "Mean"
    MAX = "Max"


PoolingMap = {
    PoolingType.FIRST: "First",
    PoolingType.LAST: "Last",
    PoolingType.MEAN: "Mean",
    PoolingType.MAX: "Max"
}


class RobertaModel:
    def __init__(self, embeddings: TorchRobertaEmbeddings,
                 encoder: BertEncoder, pooler: BertPooler,
                 config: RobertaConfig):
        self.config = config
        self.embeddings = embeddings
        self.encoder = encoder
        self.pooler = pooler
        self.prepare = cxx.PrepareBertMasks()

    def __call__(self,
                 input_ids: AnyTensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 token_type_ids: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.Tensor] = None,
                 head_mask: Optional[torch.Tensor] = None,
                 inputs_embeds: Optional[torch.Tensor] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 hidden_cache: Optional[AnyTensor] = None,
                 output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None):
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            # encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            # encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            # if encoder_attention_mask is None:
            #     encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            # encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            raise (
                "Not Implenmented self.config.is_decoder and encoder_hidden_states is not None"
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_cache = self.embeddings(input_ids=input_ids,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids)
        encoder_extended_attention_mask = try_convert(
            encoder_extended_attention_mask)
        hidden_cache = try_convert(hidden_cache)

        encoder_outputs = self.encoder(
            hidden_states=hidden_cache,
            attention_mask=encoder_extended_attention_mask,
            return_type=ReturnType.turbo_transformers)
        sequence_output = encoder_outputs[0]
        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        output = self.seq_pool(input_tensor=sequence_output,
                               return_type=return_type)
        pooler_output = self.pooler(output, return_type)
        return (
            convert_returns_as_type(sequence_output, return_type),
            pooler_output,
        ) + encoder_outputs[1:]

    @staticmethod
    def from_torch(model: TorchRobertaModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        encoder = BertEncoder.from_torch(model.encoder)
        pooler = BertPooler.from_torch(model.pooler)
        return RobertaModel(model.embeddings, encoder, pooler, model.config)

