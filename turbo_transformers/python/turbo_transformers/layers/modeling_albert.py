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
from .modeling_bert import BertEmbeddings
from .modeling_bert import BertAttention
from transformers.modeling_albert import AlbertEmbeddings as TorchAlbertEmbeddings
from transformers.modeling_albert import AlbertTransformer as TorchAlbertTransformer
from transformers.modeling_albert import AlbertAttention as TorchAlbertAttention
from transformers.modeling_albert import AlbertLayer as TorchAlbertLayer
from transformers.modeling_albert import AlbertLayerGroup as TorchAlbertLayerGroup
from transformers.modeling_albert import AlbertModel as TorchAlbertModel
from transformers.configuration_albert import AlbertConfig as TorchAlbertConfig
import torch
import enum

__all__ = [
    "AlbertEmbeddings", "AlbertTransformer", "AlbertAttention",
    "AlbertLayerGroup", "AlbertLayer", "AlbertModel", "AlbertPooler",
    "AlbertModelWithPooler"
]
ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-base': "",
    'albert-large': "",
    'albert-xlarge': "",
    'albert-xxlarge': "",
}


def _try_convert(t):
    if isinstance(t, torch.Tensor):
        return convert2tt_tensor(t)
    elif isinstance(t, np.ndarray):
        return convert2tt_tensor(torch.from_numpy(t))
    else:
        return t


def convert2tt_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))


def _to_param_dict(torch_module: torch.nn.Module):
    return {
        k: convert2tt_tensor(v)
        for k, v in torch_module.named_parameters()
    }


def _to_param_dict_naive(torch_module: torch.nn.Module):
    return {k: v for k, v in torch_module.named_parameters()}


def _create_empty_if_none(output):
    return output if output is not None else cxx.Tensor.create_empty()


AnyTensor = Union[cxx.Tensor, torch.Tensor]

class AlbertAttention(BertAttention):
    @staticmethod
    def from_torch(attention: TorchAlbertAttention):
        params = {k: v for k, v in attention.named_parameters()}

        with torch.no_grad():
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['query.weight'], params['key.weight'],
                               params['value.weight']), 0)))
            qkv_bias = torch.cat((params['query.bias'], params['key.bias'],
                                  params['value.bias']), 0)

            output_weight = torch.clone(torch.t(params['dense.weight']))
            att = AlbertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['dense.bias']),
                convert2tt_tensor(params['LayerNorm.weight']),
                convert2tt_tensor(params['LayerNorm.bias']),
                attention.num_attention_heads)

            return att

class AlbertFeedforward(cxx.AlbertFeedforward):
    def __call__(self,
                 attention_output: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None
                 ):
        attention_output = _try_convert(attention_output)
        hidden_output = cxx.Tensor.create_empty()
        output = cxx.Tensor.create_empty()
        super(AlbertFeedforward, self).__call__(attention_output, hidden_output, output)
        return convert_returns_as_type(output, return_type)

class AlbertLayer:
    def __init__(self, attention: AlbertAttention, feedforward: AlbertFeedforward):
        self.attention = attention
        self.feedforward = feedforward

    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        attention_output = cxx.Tensor.create_empty()
        attention_output, attn = self.attention(
                                        input_tensor,
                                        attention_mask,
                                        return_type=ReturnType.turbo_transformers,
                                        output=attention_output)
        output = _create_empty_if_none(output)
        output = self.feedforward(attention_output,
                                  return_type=ReturnType.turbo_transformers,
                                  output = output)
        return convert_returns_as_type(output,return_type), convert_returns_as_type(attn, return_type)

    @staticmethod
    def from_torch(layer: TorchAlbertLayer):
        layer_params = _to_param_dict_naive(layer)

        weight = torch.clone(torch.t(layer_params["ffn.weight"]))
        weight_output = torch.clone(
            torch.t(layer_params["ffn_output.weight"]))
        return AlbertLayer(
            AlbertAttention.from_torch(layer.attention),
            AlbertFeedforward(convert2tt_tensor(weight),
                              convert2tt_tensor(layer_params['ffn.bias']),
                              convert2tt_tensor(weight_output),
                              convert2tt_tensor(layer_params['ffn_output.bias']),
                              convert2tt_tensor(layer_params['full_layer_layer_norm.weight']),
                              convert2tt_tensor(layer_params['full_layer_layer_norm.bias'])
                              )
        )


class AlbertLayerGroup:
    def __init__(self, layer: Sequence[AlbertLayer]):
        self.layer = layer

    @staticmethod
    def from_torch(encoder: TorchAlbertLayerGroup):
        layer = [
            AlbertLayer.from_torch(albert_layer)
            for albert_layer in encoder.albert_layers
        ]
        return AlbertLayerGroup(layer)

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        output = _create_empty_if_none(output)
        first = True
        for l in self.layer:
            if first:
                input_states = hidden_states
                first = False
            else:
                input_states = output
            output, _ = l(input_tensor=input_states,
                          attention_mask=attention_mask,
                          return_type=ReturnType.turbo_transformers,
                          output=output)
        return (convert_returns_as_type(output, return_type), )



class AlbertTransformer(cxx.AlbertTransformer):
    def __init__(self, group: Sequence[AlbertLayerGroup], weights, bias, cfg):
        self.group = group
        self.cfg = cfg
        super(AlbertTransformer, self).__init__(weights, bias)

    @staticmethod
    def from_torch(transformer: TorchAlbertTransformer, cfg):
        params = _to_param_dict_naive(transformer)
        weights = torch.clone(
            torch.t(params["embedding_hidden_mapping_in.weight"]))
        group = [
            AlbertLayerGroup.from_torch(albert_group)
            for albert_group in transformer.albert_layer_groups
        ]
        return AlbertTransformer(group, convert2tt_tensor(weights), \
                  convert2tt_tensor(params['embedding_hidden_mapping_in.bias']),\
                  cfg)


    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        hidden_states = _try_convert(hidden_states)
        output = _create_empty_if_none(output)
        super(AlbertTransformer, self).__call__(hidden_states, output)
        hidden_states = _try_convert(output)
        output = cxx.Tensor.create_empty()
        first = True
        for i in range(self.cfg.num_hidden_layers):
            group_idx = int(
                i / (self.cfg.num_hidden_layers / self.cfg.num_hidden_groups))

            if first:
                input_states = hidden_states
                first = False
            else:
                input_states = output
            output = self.group[group_idx](\
                hidden_states = input_states, \
                return_type = ReturnType.turbo_transformers,\
                attention_mask = attention_mask,\
                output = output)[0]

        return convert_returns_as_type(output, return_type)

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

class AlbertPooler(cxx.BertPooler):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output = _create_empty_if_none(output)
        super(AlbertPooler, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(pooler: torch.nn.Module):
        pooler_params = {k:v for k,v in pooler.named_parameters()}
        weight = torch.clone(torch.t(pooler_params['weight']))
        return AlbertPooler(convert2tt_tensor(weight),
                          convert2tt_tensor(pooler_params['bias']))


class SequencePool(cxx.SequencePool):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output_tensor: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output_tensor = _create_empty_if_none(output_tensor)
        super(SequencePool, self).__call__(input_tensor, output_tensor)
        return convert_returns_as_type(output_tensor, return_type)


class AlbertModel:
    def __init__(self, Embeddings: BertEmbeddings, Encoder: AlbertTransformer):
        self.embedding = Embeddings
        self.encoder = Encoder
        self.prepare = cxx.PrepareBertMasks()


    def __call__(self, inputs: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 hidden_cache: Optional[AnyTensor] = None,
                 output: Optional[AnyTensor] = None,
                 return_type: Optional[AnyTensor] = None):
        attention_mask = _try_convert(_create_empty_if_none(attention_mask))
        token_type_ids = _try_convert(_create_empty_if_none(token_type_ids))
        position_ids = _try_convert(_create_empty_if_none(position_ids))
        inputs = _try_convert(inputs)
        extended_attention_masks = cxx.Tensor.create_empty()
        output = _create_empty_if_none(output)
        hidden_cache = _create_empty_if_none(hidden_cache)
        embedding_cache = cxx.Tensor.create_empty()

        self.prepare(inputs, attention_mask, token_type_ids, position_ids,
                     extended_attention_masks)

        embedding_cache = self.embedding(inputs,
                                      position_ids = position_ids,
                                      token_type_ids = token_type_ids,
                                      output = embedding_cache,
                                      return_type = ReturnType.turbo_transformers)

        hidden_cache = self.encoder(hidden_states = embedding_cache,
                                    attention_mask = extended_attention_masks,
                                    return_type = ReturnType.turbo_transformers,
                                    output = hidden_cache)

        self.seq_pool = SequencePool(PoolingMap[pooling_type])

        output = self.seq_pool(input_tensor=hidden_cache,
                               return_type=return_type,
                               output_tensor=output)

        return output, convert_returns_as_type(hidden_cache, return_type)

    @staticmethod
    def from_torch(model: TorchAlbertModel, cfg: TorchAlbertConfig, device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = AlbertTransformer.from_torch(model.encoder, cfg)
        return AlbertModel(embeddings, encoder)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchAlbertModel.from_pretrained(model_id_or_path)
        model = AlbertModel.from_torch(torch_model, torch_model.config, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model



class AlbertModelWithPooler:
    def __init__(self, albertmodel: AlbertModel, pooler: AlbertPooler):
        self.albertmodel = albertmodel
        self.pooler = pooler

    def __call__(self,
                 inputs: AnyTensor,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 hidden_cache: Optional[AnyTensor] = None,
                 pooler_output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None):
        encoder_output, hidden_cache = self.albertmodel(
            inputs,
            attention_mask = attention_masks,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            pooling_type = pooling_type,
            hidden_cache = hidden_cache,
            output=None,
            return_type=ReturnType.turbo_transformers)
        pooler_output = self.pooler(encoder_output, return_type, pooler_output)
        return pooler_output, convert_returns_as_type(encoder_output, return_type),\
               convert_returns_as_type(hidden_cache, return_type)

    @staticmethod
    def from_torch(model: TorchAlbertModel, cfg: TorchAlbertConfig,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = AlbertTransformer.from_torch(model.encoder, cfg)
        albertmodel = AlbertModel(embeddings, encoder)
        pooler = AlbertPooler.from_torch(model.pooler)
        return AlbertModelWithPooler(albertmodel, pooler)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchAlbertModel.from_pretrained(model_id_or_path)
        model = AlbertModelWithPooler.from_torch(torch_model, torch_model.config, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model
