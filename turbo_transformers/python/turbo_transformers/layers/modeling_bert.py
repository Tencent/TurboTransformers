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
import torch.utils.dlpack as dlpack

from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.modeling_bert import BertOutput as TorchBertOutput
from transformers.modeling_bert import BertAttention as TorchBertAttention
from transformers.modeling_bert import BertLayer as TorchBertLayer
from transformers.modeling_bert import BertEncoder as TorchBertEncoder
from transformers.modeling_bert import BertModel as TorchBertModel
from transformers.modeling_bert import BertPooler as TorchBertPooler

from onmt.modules.multi_headed_attn import MultiHeadedAttention as OnmtMultiHeadedAttention

import enum
import numpy as np

__all__ = [
    'BertEmbeddings', 'BertIntermediate', 'BertOutput', 'BertAttention',
    'BertLayer', 'BertEncoder', 'SequencePool', 'BertModel', 'PoolingType',
    'BertPooler', 'BertModelWithPooler', 'MultiHeadedAttention'
]


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


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self,
                 input_ids: AnyTensor,
                 position_ids: AnyTensor,
                 token_type_ids: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_ids = _try_convert(input_ids)
        position_ids = _try_convert(position_ids)
        token_type_ids = _try_convert(token_type_ids)
        output = _create_empty_if_none(output)
        super(BertEmbeddings, self).__call__(input_ids, position_ids,
                                             token_type_ids, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(bert_embedding: TorchBertEmbeddings) -> 'BertEmbeddings':
        params = _to_param_dict(bert_embedding)
        return BertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'])

    @staticmethod
    def from_npz(file_name: str):
        f = np.load(file_name)
        return BertEmbeddings(
            _try_convert(f['embeddings.word_embeddings.weight']),
            _try_convert(f['embeddings.position_embeddings.weight']),
            _try_convert(f['embeddings.token_type_embeddings.weight']),
            _try_convert(f['embeddings.LayerNorm.weight']),
            _try_convert(f['embeddings.LayerNorm.bias']))


class BertIntermediate(cxx.BertIntermediate):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output = _create_empty_if_none(output)
        super(BertIntermediate, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = _to_param_dict_naive(intermediate)
        weight = torch.clone(torch.t(intermediate_params["dense.weight"]))
        return BertIntermediate(
            convert2tt_tensor(weight),
            convert2tt_tensor(intermediate_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str, layer_num: int):
        f = np.load(file_name)
        return BertIntermediate(
            _try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.weight']),
            _try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.bias']))


class BertOutput(cxx.BertOutput):
    def __call__(self,
                 intermediate_output: AnyTensor,
                 attention_output: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        intermediate_output = _try_convert(intermediate_output)
        attention_output = _try_convert(attention_output)
        output = _create_empty_if_none(output)
        super(BertOutput, self).__call__(intermediate_output, attention_output,
                                         output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(output: TorchBertOutput):
        params = _to_param_dict_naive(output)
        weight = convert2tt_tensor(torch.clone(torch.t(
            params["dense.weight"])))
        return BertOutput(weight, convert2tt_tensor(params["dense.bias"]),
                          convert2tt_tensor(params["LayerNorm.weight"]),
                          convert2tt_tensor(params["LayerNorm.bias"]))

    @staticmethod
    def from_npz(file_name: str, layer_num: int):
        f = np.load(file_name)
        return BertOutput(
            _try_convert(f[f'encoder.layer.{layer_num}.output.dense.weight']),
            _try_convert(f[f'encoder.layer.{layer_num}.output.dense.bias']),
            _try_convert(
                f[f'encoder.layer.{layer_num}.output.LayerNorm.weight']),
            _try_convert(
                f[f'encoder.layer.{layer_num}.output.LayerNorm.bias']))


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
        key_tensor = _try_convert(key_tensor)
        value_tensor = _try_convert(value_tensor)
        query_tensor = _try_convert(query_tensor)
        mask_tensor = _try_convert(mask_tensor)
        # assert layer_cache == None
        # assert attn_type == "context"
        output = _create_empty_if_none(output)
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


class BertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        attention_mask = _try_convert(attention_mask)
        output = _create_empty_if_none(output)
        super(BertAttention, self).__call__(input_tensor, attention_mask,
                                            output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(attention: TorchBertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['self.query.weight'],
                               params['self.key.weight'],
                               params['self.value.weight']), 0)))
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0)

            output_weight = torch.clone(torch.t(params['output.dense.weight']))
            att = BertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['output.dense.bias']),
                convert2tt_tensor(params['output.LayerNorm.weight']),
                convert2tt_tensor(params['output.LayerNorm.bias']),
                attention.self.num_attention_heads)

            return att

    @staticmethod
    def from_npz(file_name: str, layer_num: int, num_attention_heads: int):
        f = np.load(file_name)
        return BertAttention(
            _try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.weight']),
            _try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.bias']),
            _try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.weight']),
            _try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.bias']),
            _try_convert(f[
                f'encoder.layer.{layer_num}.attention.output.LayerNorm.weight']
                         ),
            _try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.bias']
            ), num_attention_heads)


class BertLayer:
    def __init__(self, attention: BertAttention,
                 intermediate: BertIntermediate, output: BertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 attention_output: Optional[cxx.Tensor] = None,
                 intermediate_output: Optional[cxx.Tensor] = None,
                 output: Optional[cxx.Tensor] = None):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            return_type=ReturnType.turbo_transformers,
            output=attention_output)
        intermediate_output = self.intermediate(
            attention_output,
            return_type=ReturnType.turbo_transformers,
            output=intermediate_output)
        return self.output(intermediate_output,
                           attention_output,
                           return_type=return_type,
                           output=output)

    @staticmethod
    def from_torch(layer: TorchBertLayer):
        return BertLayer(BertAttention.from_torch(layer.attention),
                         BertIntermediate.from_torch(layer.intermediate),
                         BertOutput.from_torch(layer.output))

    @staticmethod
    def from_npz(file_name: str, layer_num: int, num_attention_heads: int):
        f = np.load(file_name)
        return BertLayer(
            BertAttention.from_npz(file_name, layer_num, num_attention_heads),
            BertIntermediate.from_npz(file_name, layer_num),
            BertOutput.from_npz(file_name, layer_num))


class BertEncoder:
    def __init__(self, layer: Sequence[BertLayer]):
        self.layer = layer

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 attention_output: Optional[cxx.Tensor] = None,
                 intermediate_output: Optional[cxx.Tensor] = None,
                 output: Optional[cxx.Tensor] = None):
        attention_output = _create_empty_if_none(attention_output)
        intermediate_output = _create_empty_if_none(intermediate_output)
        output = _create_empty_if_none(output)
        first = True
        for l in self.layer:
            if first:
                input_states = hidden_states
                first = False
            else:
                input_states = output

            output = l(hidden_states=input_states,
                       attention_mask=attention_mask,
                       return_type=ReturnType.turbo_transformers,
                       attention_output=attention_output,
                       intermediate_output=intermediate_output,
                       output=output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(encoder: TorchBertEncoder):
        layer = [
            BertLayer.from_torch(bert_layer) for bert_layer in encoder.layer
        ]
        return BertEncoder(layer)

    @staticmethod
    def from_npz(file_name: str, num_hidden_layers: int,
                 num_attention_heads: int):
        layer = []
        for i in range(num_hidden_layers):
            layer.append(BertLayer.from_npz(file_name, i, num_attention_heads))
        return BertEncoder(layer)


class SequencePool(cxx.SequencePool):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output_tensor: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output_tensor = _create_empty_if_none(output_tensor)
        super(SequencePool, self).__call__(input_tensor, output_tensor)
        return convert_returns_as_type(output_tensor, return_type)


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


class BertModel:
    def __init__(self, embeddings: BertEmbeddings, encoder: BertEncoder):
        self.embeddings = embeddings
        self.encoder = encoder
        self.prepare = cxx.PrepareBertMasks()

    def __call__(self,
                 inputs: AnyTensor,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 hidden_cache: Optional[AnyTensor] = None,
                 output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None):
        attention_masks = _try_convert(_create_empty_if_none(attention_masks))
        token_type_ids = _try_convert(_create_empty_if_none(token_type_ids))
        position_ids = _try_convert(_create_empty_if_none(position_ids))
        inputs = _try_convert(inputs)
        extended_attention_masks = cxx.Tensor.create_empty()
        output = _create_empty_if_none(output)
        hidden_cache = _create_empty_if_none(hidden_cache)

        self.prepare(inputs, attention_masks, token_type_ids, position_ids,
                     extended_attention_masks)

        hidden_cache = self.embeddings(
            inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output=hidden_cache,
            return_type=ReturnType.turbo_transformers)

        hidden_cache = self.encoder(hidden_states=hidden_cache,
                                    attention_mask=extended_attention_masks,
                                    return_type=ReturnType.turbo_transformers,
                                    output=hidden_cache)

        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        output = self.seq_pool(input_tensor=hidden_cache,
                               return_type=return_type,
                               output_tensor=output)
        return output, convert_returns_as_type(hidden_cache, return_type)

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        return BertModel(embeddings, encoder)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModel.from_torch(torch_model, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        embeddings = BertEmbeddings.from_npz(file_name)
        encoder = BertEncoder.from_npz(file_name, config.num_hidden_layers,
                                       config.num_attention_heads)
        return BertModel(embeddings, encoder)


class BertPooler(cxx.BertPooler):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output = _create_empty_if_none(output)
        super(BertPooler, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(pooler: TorchBertPooler):
        pooler_params = _to_param_dict_naive(pooler)
        weight = torch.clone(torch.t(pooler_params['dense.weight']))
        return BertPooler(convert2tt_tensor(weight),
                          convert2tt_tensor(pooler_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str, device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertPooler(_try_convert(f['pooler.dense.weight']),
                          _try_convert(f['pooler.dense.bias']))


class BertModelWithPooler:
    def __init__(self, bertmodel: BertModel, pooler: BertPooler):
        self.bertmodel = bertmodel
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
        encoder_output, hidden_cache = self.bertmodel(
            inputs,
            attention_masks,
            token_type_ids,
            position_ids,
            pooling_type,
            hidden_cache,
            output=None,
            return_type=ReturnType.turbo_transformers)
        pooler_output = self.pooler(encoder_output, return_type, pooler_output)
        return pooler_output, convert_returns_as_type(
            encoder_output,
            return_type), convert_returns_as_type(hidden_cache, return_type)

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        bertmodel = BertModel(embeddings, encoder)
        pooler = BertPooler.from_torch(model.pooler)
        return BertModelWithPooler(bertmodel, pooler)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModelWithPooler.from_torch(torch_model, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        model = BertModel.from_npz(file_name, config)
        pooler = BertPooler.from_npz(file_name)
        return BertModelWithPooler(model, pooler)
