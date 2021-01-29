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

import torch
import torch.utils.dlpack as dlpack
from torch import Tensor, device, dtype, nn
from typing import Union
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from .return_type import convert_returns_as_type, ReturnType

__all__ = [
    'try_convert', 'convert2tt_tensor', 'to_param_dict_convert_tt',
    'to_param_dict', 'create_empty_if_none', 'AnyTensor'
]


def convert2tt_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))


def try_convert(t, device: Optional[torch.device] = None):
    if isinstance(t, torch.Tensor):
        return convert2tt_tensor(t)
    elif isinstance(t, np.ndarray):
        if device is not None:
            return convert2tt_tensor(torch.from_numpy(t).to(device))
        else:
            return convert2tt_tensor(torch.from_numpy(t))
    else:
        return t


def to_param_dict_convert_tt(torch_module: torch.nn.Module):
    return {
        k: convert2tt_tensor(v)
        for k, v in torch_module.named_parameters()
    }


def to_param_dict(torch_module: torch.nn.Module):
    return {k: v for k, v in torch_module.named_parameters()}


def create_empty_if_none(output):
    return output if output is not None else cxx.Tensor.create_empty()


def get_head_mask(head_mask,
                  num_hidden_layers: int,
                  is_attention_chunked: bool = False):
    """
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    attention_probs has shape bsz x n_heads x N x N
    Arguments:
        head_mask: torch.Tensor or None: has shape [num_heads] or [num_hidden_layers x num_heads]
        num_hidden_layers: int
    Returns:
            Tensor of shape shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            or list with [None] for each layer
    """
    if head_mask is not None:
        head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers

    return head_mask


def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple,
                                device: device) -> Tensor:
    """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.
    Arguments:
        attention_mask: torch.Tensor with 1 indicating tokens to ATTEND to
        input_shape: tuple, shape of input_ids
        device: torch.Device, usually self.device
    Returns:
        torch.Tensor with dtype of attention_mask.dtype
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})"
            .format(input_shape, attention_mask.shape))

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


AnyTensor = Union[cxx.Tensor, torch.Tensor]
