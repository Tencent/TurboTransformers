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
from .modules import MultiHeadedAttention, PositionwiseFeedForward

from .utils import try_convert, convert2tt_tensor, create_empty_if_none, AnyTensor

from onmt.encoders.transformer import TransformerEncoder as OnmtTransformerEncoder
from onmt.encoders.transformer import TransformerEncoderLayer as OnmtTransformerEncoderLayer
from onmt.encoders.encoder import EncoderBase

from torch.nn import LayerNorm as TorchLayerNorm
from onmt.utils.misc import sequence_mask
import torch.nn as nn
import enum
import numpy as np

__all__ = [
    'TransformerEncoderLayer', 'TransformerEncoder'
]


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, self_attn: MultiHeadedAttention, 
                feed_forward: PositionwiseFeedForward):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

    @classmethod
    def from_onmt(cls, encoder_layer: OnmtTransformerEncoderLayer):
        self_attn = MultiHeadedAttention.from_onmt(encoder_layer.self_attn,
                                                encoder_layer.layer_norm)
        feed_forward = PositionwiseFeedForward.from_onmt(encoder_layer.feed_forward)
        return cls(self_attn, feed_forward)

    def __call__(self, inputs, mask, return_type: Optional[ReturnType] = None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_tensor = try_convert(inputs)
        mask_tensor = try_convert(mask.float())

        context, _ = self.self_attn(
                            input_tensor, 
                            input_tensor, 
                            input_tensor,
                            mask=mask_tensor, 
                            attn_type="self",
                            pre_layernorm=True,
                            post_add_input=True,
                            return_type=return_type)

        return self.feed_forward(context, return_type=return_type)



class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, embeddings, layers, layer_norm: TorchLayerNorm):
        super(TransformerEncoder, self).__init__()
        self.embeddings = embeddings
        self.transformer = layers
        self.layer_norm = layer_norm

    @classmethod
    def from_onmt(cls, onmt_encoder: OnmtTransformerEncoder, 
                device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            onmt_encoder.to(device)
        layers = [
            TransformerEncoderLayer.from_onmt(layer)
            for layer in onmt_encoder.transformer
        ]
        return cls(onmt_encoder.embeddings, layers, onmt_encoder.layer_norm)


    def __call__(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths