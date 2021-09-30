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
from typing import Union, Optional, Sequence, List
import torch
from .return_type import convert_returns_as_type, ReturnType

from .utils import try_convert, convert2tt_tensor, create_empty_if_none, AnyTensor

from onmt.modules.multi_headed_attn import MultiHeadedAttention as OnmtMultiHeadedAttention

from transformers.models.bert.modeling_bert import BertAttention as TorchBertAttention
from transformers.models.bert.modeling_bert import BertLayer as TorchBertLayer
from transformers.models.bert.modeling_bert import BertEncoder as TorchBertEncoder
from transformers.models.bert.modeling_bert import BertModel as TorchBertModel
from transformers.models.bert.modeling_bert import BertConfig as TorchBertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings

from .modeling_bert import BertIntermediate
from .modeling_bert import BertOutput
from .modeling_bert import BertEmbeddings
from .modeling_bert import BertEncoder
from .modeling_bert import BertPooler
from .modeling_bert import SequencePool
from .modeling_bert import PoolingMap
from .modeling_bert import PoolingType

from torch.nn import LayerNorm as TorchLayerNorm

import numpy as np

__all__ = [
    'MultiHeadedAttentionSmartBatch', 'BertLayerSmartBatch',
    'BertEncoderSmartBatch', 'BertModelSmartBatch'
]


class MultiHeadedAttentionSmartBatch(cxx.MultiHeadedAttentionSmartBatch):
    def __call__(self,
                 key_tensor: AnyTensor,
                 value_tensor: AnyTensor,
                 query_tensor: AnyTensor,
                 query_seq_len_list: Sequence[int],
                 key_seq_len_list: Sequence[int],
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
        """ Implement a MultiHeadedAttention with SmartBatch
        https://github.com/bytedance/effective_transformer
        Additional Parameter:
        @query_seq_len_list contains a list of input_seq_len.
        """

        # TODO(jiaruifang) bug device is only cpu, you must make it support GPU
        if mask is None:
            # (B, 1, k_len)
            if attn_type == "self":
                # self attn. query_seq_len is the same as the key_seq_len
                batch_size = len(query_seq_len_list)
                query_max_seq_len = max(query_seq_len_list)
                if isinstance(query_tensor, torch.Tensor):
                    mask = torch.zeros(batch_size,
                                       1,
                                       query_max_seq_len,
                                       dtype=torch.float32,
                                       device=query_tensor.device)
                else:
                    raise "Mask is None and MultiHeadedAttentionSmartBatch can not identify the device type of mask"
                for batch_idx in range(batch_size):
                    for query_seq_idx in range(query_seq_len_list[batch_idx],
                                               query_max_seq_len):
                        mask[batch_idx][0][query_seq_idx] = -1e9
            elif attn_type == "context":
                batch_size = len(query_seq_len_list)
                assert (batch_size == len(key_seq_len_list))
                query_max_seq_len = max(query_seq_len_list)
                key_max_seq_len = max(key_seq_len_list)
                if isinstance(query_tensor, torch.Tensor):
                    mask = torch.zeros(batch_size,
                                       1,
                                       key_max_seq_len,
                                       dtype=torch.float32,
                                       device=query_tensor.device)
                else:
                    raise "Mask is None and MultiHeadedAttentionSmartBatch can not identify the device type of mask"
                for batch_idx in range(batch_size):
                    for key_seq_idx in range(key_seq_len_list[batch_idx],
                                             key_max_seq_len):
                        mask[batch_idx][0][key_seq_idx] = -1e9

        mask = try_convert(mask)
        key_tensor = try_convert(key_tensor)
        value_tensor = try_convert(value_tensor)
        query_tensor = try_convert(query_tensor)

        output = create_empty_if_none(output)
        attn = create_empty_if_none(attn)
        layer_cache_tmp = {}
        if layer_cache is not None:
            for k, v in layer_cache.items():
                if v is not None:
                    layer_cache_tmp[k] = try_convert(v)
                else:
                    layer_cache_tmp[k] = create_empty_if_none(v)

        super(MultiHeadedAttentionSmartBatch,
              self).__call__(key_tensor, value_tensor, query_tensor, mask,
                             attn_type, output, attn, layer_cache_tmp,
                             query_seq_len_list, key_seq_len_list,
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
    def pack_parameter(attn_params: dict,
                       is_trans_weight: Optional[bool] = False):
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
            att = MultiHeadedAttentionSmartBatch(
                *(MultiHeadedAttentionSmartBatch.pack_parameter(
                    attn_params, is_trans_weight)),
                multi_headed_attn.head_count)
            return att

    #
    # @staticmethod
    # def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
    #               layer_norm: TorchLayerNorm,
    #               is_trans_weight: bool = False):
    #     ln_params = {k: v for k, v in layer_norm.named_parameters()}
    #     attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
    #     with torch.no_grad():
    #         att = MultiHeadedAttentionSmartBatch(
    #             *(MultiHeadedAttention.pack_parameter(multi_headed_attn,
    #                                                   is_trans_weight)),
    #             convert2tt_tensor(ln_params['weight']),
    #             convert2tt_tensor(ln_params['bias']),
    #             multi_headed_attn.head_count)
    #         return att

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
                att = MultiHeadedAttentionSmartBatch(
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
                att = MultiHeadedAttentionSmartBatch(
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


class BertLayerSmartBatch:
    def __init__(self, attention: MultiHeadedAttentionSmartBatch,
                 intermediate: BertIntermediate, output: BertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: AnyTensor,
                 query_seq_len_list: Sequence,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions=False,
                 return_type: Optional[ReturnType] = None):
        #self_attention_outputs[0] (1, sum_from_seq_len, hidden)
        self_attention_outputs = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            query_seq_len_list, [],
            mask=attention_mask,
            layer_cache=None,
            attn_type="self",
            post_layernorm=True,
            return_type=ReturnType.turbo_transformers)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(
            attention_output, return_type=ReturnType.turbo_transformers)
        layer_output = self.output(intermediate_output,
                                   attention_output,
                                   return_type=return_type)
        outputs = (layer_output, ) + outputs
        return outputs

    @staticmethod
    def from_torch(layer: TorchBertLayer):
        return BertLayerSmartBatch(
            MultiHeadedAttentionSmartBatch.from_torch(layer.attention),
            BertIntermediate.from_torch(layer.intermediate),
            BertOutput.from_torch(layer.output))

    @staticmethod
    def from_npz(file_name: str, layer_num: int, num_attention_heads: int):
        f = np.load(file_name)
        return BertLayerSmartBatch(
            BertAttention.from_npz(file_name, layer_num, num_attention_heads),
            BertIntermediate.from_npz(file_name, layer_num),
            BertOutput.from_npz(file_name, layer_num))


class BertEncoderSmartBatch:
    def __init__(self, layer: Sequence[BertLayerSmartBatch]):
        self.layer = layer

    def __call__(self,
                 hidden_states: AnyTensor,
                 query_seq_len_list: Sequence,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None):
        all_hidden_states = ()
        all_attentions = ()
        hidden_states = try_convert(hidden_states)
        for l in self.layer:
            layer_outputs = l(hidden_states=hidden_states,
                              query_seq_len_list=query_seq_len_list,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions,
                              return_type=ReturnType.turbo_transformers)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        outputs = (convert_returns_as_type(hidden_states, return_type), )
        # Add last layer
        if output_hidden_states:
            # TODO(jiaruifang)two return value use the same memory space, that is not supported in dlpack.
            # So we do not append the last hidden_state at the buttom of all_hidden_states,
            # User should use outputs[0] if necessary
            # all_hidden_states = all_hidden_states + (convert_returns_as_type(hidden_states, ReturnType.TORCH),)
            pass

        if output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if output_attentions:
            outputs = outputs + (all_attentions, )

        return outputs

    @staticmethod
    def from_torch(encoder: TorchBertEncoder):
        layer = [
            BertLayerSmartBatch.from_torch(bert_layer)
            for bert_layer in encoder.layer
        ]
        return BertEncoderSmartBatch(layer)

    @staticmethod
    def from_npz(file_name: str, num_hidden_layers: int,
                 num_attention_heads: int):
        layer = []
        for i in range(num_hidden_layers):
            layer.append(
                BertLayerSmartBatch.from_npz(file_name, i,
                                             num_attention_heads))
        return BertEncoderSmartBatch(layer)


class BertModelNoPoolerSmartBatch:
    def __init__(self, embeddings: TorchBertEmbeddings,
                 encoder: BertEncoderSmartBatch):
        self.embeddings = embeddings  #torch
        self.encoder = encoder
        self.prepare = cxx.PrepareBertMasks()

    def __call__(
            self,
            inputs: Sequence[torch.Tensor],
            query_seq_len_list: Sequence,
            attention_masks: Optional[AnyTensor] = None,
            token_type_ids: Optional[AnyTensor] = None,
            position_ids: Optional[AnyTensor] = None,
            head_mask: Optional[AnyTensor] = None,
            inputs_embeds: Optional[AnyTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            pooling_type: PoolingType = PoolingType.
            FIRST,  #the following parameters are exclusive for turbo
            return_type: Optional[ReturnType] = None):
        # serially embed the inputs
        # TODO position_ids and token_type_ids should be a list
        Q_list = []
        for input in inputs:
            embedding_output = self.embeddings(input_ids=input,
                                               position_ids=position_ids,
                                               token_type_ids=token_type_ids)
            Q_list.append(embedding_output)
        # concat Qs together
        for idx, Q in enumerate(Q_list):
            if idx == 0:
                self.concat_Q = Q
            else:
                self.concat_Q = torch.cat((self.concat_Q, Q), 1)
        query_max_seq_len = max(query_seq_len_list)
        batch_size = len(query_seq_len_list)
        assert (batch_size != 0)
        if attention_masks is None:
            mask = torch.zeros(batch_size,
                               1,
                               query_max_seq_len,
                               dtype=torch.float32,
                               device=inputs[0].device)
            for batch_idx in range(batch_size):
                for query_seq_idx in range(query_seq_len_list[batch_idx],
                                           query_max_seq_len):
                    mask[batch_idx][0][query_seq_idx] = -1e9
            mask = try_convert(mask)
        else:
            mask = try_convert(attention_masks)

        encoder_outputs = self.encoder(
            hidden_states=self.concat_Q,
            query_seq_len_list=query_seq_len_list,
            attention_mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_type=return_type)
        return encoder_outputs

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        # embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoderSmartBatch.from_torch(model.encoder)
        return BertModelNoPoolerSmartBatch(model.embeddings, encoder)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModelNoPoolerSmartBatch.from_torch(torch_model, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        embeddings = BertEmbeddings.from_npz(file_name)
        encoder = BertEncoderSmartBatch.from_npz(file_name,
                                                 config.num_hidden_layers,
                                                 config.num_attention_heads)
        return BertModelNoPoolerSmartBatch(embeddings, encoder)


class BertModelSmartBatch:
    def __init__(self, model: BertModelNoPoolerSmartBatch, pooler: BertPooler,
                 config: TorchBertConfig):
        self.config = config
        self.bertmodel_nopooler = model
        self.pooler = pooler

    def __call__(self,
                 inputs: AnyTensor,
                 query_seq_len_list: Sequence,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 inputs_embeds: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 pooling_type: PoolingType = PoolingType.FIRST,
                 pooler_output: Optional[AnyTensor] = None,
                 return_type: Optional[ReturnType] = None):
        encoder_outputs = self.bertmodel_nopooler(
            inputs,
            query_seq_len_list,
            attention_masks,
            token_type_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pooling_type=pooling_type,
            return_type=ReturnType.turbo_transformers)

        sequence_output = encoder_outputs[0]
        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        sequence_pool_output = self.seq_pool(
            input_tensor=sequence_output,
            return_type=ReturnType.turbo_transformers)
        pooler_output = self.pooler(sequence_pool_output, return_type,
                                    pooler_output)
        return (
            convert_returns_as_type(sequence_output, return_type),
            pooler_output,
        ) + encoder_outputs[1:]

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        # use_gpu = False
        # if device is None:
        #     device = model.device
        # # we may need to move to GPU explicitly
        # if 'cuda' in device.type and torch.cuda.is_available():
        #     model.to(device)
        #     if backend is None:
        #         backend = "turbo"  # On GPU turbo is faster
        #     use_gpu = True

        # embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoderSmartBatch.from_torch(model.encoder)
        bertmodel_nopooler = BertModelNoPoolerSmartBatch(
            model.embeddings, encoder)
        pooler = BertPooler.from_torch(model.pooler)
        return BertModelSmartBatch(bertmodel_nopooler, pooler, model.config)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModelSmartBatch.from_torch(torch_model, device,
                                               torch_model.config)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config):
        model = BertModelNoPoolerSmartBatch.from_npz(file_name, config)
        pooler = BertPooler.from_npz(file_name)
        return BertModelSmartBatch(model, pooler)
