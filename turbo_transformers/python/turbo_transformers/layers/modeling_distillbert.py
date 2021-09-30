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

from transformers.models.distilbert.modeling_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention as TorchDistilMultiHeadSelfAttention
from transformers.models.distilbert.modeling_distilbert import FFN as TorchDistilFFN
from transformers.models.distilbert.modeling_distilbert import TransformerBlock as TorchDistilTransformerBlock
from transformers.models.distilbert.modeling_distilbert import Transformer as TorchDistilTransformer
from transformers.models.distilbert.modeling_distilbert import Embeddings as TorchDistrilEmbeddings
from transformers.models.distilbert.modeling_distilbert import DistilBertModel as TorchDistilBertModel

from torch import nn
import numpy as np
__all__ = [
    'DistillBertAttention', 'DistrillFFN', 'DistrillTransformerBlock',
    'DistrillTransformer', 'DistilBertModel'
]


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
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (
                1.0 - attention_mask) * -10000.0  #-float("inf") will cause NAN

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


class DistrillFFN(cxx.DistrillFFN):
    def __call__(
            self,
            input_tensor: AnyTensor,
            return_type: Optional[ReturnType] = None,
            is_trans_weight: Optional[bool] = True,  #Intel 61xx True is faster
            output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(DistrillFFN, self).__call__(input_tensor, output,
                                          is_trans_weight)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(ffn: TorchDistilFFN,
                   layernorm: nn.LayerNorm,
                   is_trans_weight: Optional[bool] = True):
        ffn_params = {k: v for k, v in ffn.named_parameters()}
        layernorm_params = {k: v for k, v in layernorm.named_parameters()}

        # Note that torch's weights of linear layer is transposed
        if is_trans_weight:
            w_1 = convert2tt_tensor(ffn_params['lin1.weight'])
            w_2 = convert2tt_tensor(ffn_params['lin2.weight'])
        else:
            w_1 = convert2tt_tensor(
                torch.clone(torch.t(ffn_params['lin1.weight']).contiguous()))
            w_2 = convert2tt_tensor(
                torch.clone(torch.t(ffn_params['lin2.weight']).contiguous()))

        with torch.no_grad():
            ffn = DistrillFFN(w_1, convert2tt_tensor(ffn_params['lin1.bias']),
                              w_2, convert2tt_tensor(ffn_params['lin2.bias']),
                              convert2tt_tensor(layernorm_params['weight']),
                              convert2tt_tensor(layernorm_params['bias']))
            return ffn


class DistrillTransformerBlock:
    def __init__(self, attn: DistillBertAttention, ffn: DistrillFFN):
        self.attention = attn
        self.ffn = ffn

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 head_mask: Optional[torch.Tensor] = None,
                 output_attentions=False,
                 return_type: Optional[ReturnType] = None):
        hidden_states = try_convert(hidden_states)

        sa_output = self.attention(hidden_states,
                                   attention_mask,
                                   head_mask,
                                   output_attentions=output_attentions,
                                   return_type=ReturnType.turbo_transformers)
        if output_attentions:
            sa_output, sa_weights = sa_output
        else:
            sa_output = sa_output[0]
        ffn_output = self.ffn(sa_output)
        output = (ffn_output, )
        if output_attentions:
            output = (sa_weights, ) + output
        return output

    @staticmethod
    def from_torch(layer: TorchDistilTransformerBlock):
        return DistrillTransformerBlock(
            DistillBertAttention.from_torch(layer.attention,
                                            layer.sa_layer_norm),
            DistrillFFN.from_torch(layer.ffn, layer.output_layer_norm))


class DistrillTransformer:
    def __init__(self, blocks: Sequence[DistrillTransformerBlock]):
        self.blocks = blocks

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 return_type: Optional[ReturnType] = ReturnType.TORCH):
        all_hidden_states = ()
        all_attentions = ()
        hidden_states = try_convert(hidden_states)
        for l in self.blocks:
            layer_outputs = l(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions,
                              return_type=ReturnType.turbo_transformers)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # outputs = (convert_returns_as_type(hidden_states, return_type), )
        outputs = (hidden_states, )
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
    def from_torch(transform: TorchDistilTransformer):
        blocks = [
            DistrillTransformerBlock.from_torch(l) for l in transform.layer
        ]
        return DistrillTransformer(blocks)


class DistilBertModel:
    def __init__(self,
                 embeddings_onnxmodel_variant,
                 transformer: DistrillTransformer,
                 backend="turbo"):
        if backend == "turbo":
            self.embeddings = embeddings_onnxmodel_variant
            self.transformer = transformer
            self.backend = "turbo"
        elif backend == "onnxrt":
            self.onnxmodel = embeddings_onnxmodel_variant
            self.backend = "onnxrt"

    def __call__(self,
                 input_ids: AnyTensor,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 inputs_embeds: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = None,
                 output_hidden_states: Optional[bool] = None,
                 return_type: Optional[ReturnType] = None):
        if self.backend == "onnxrt":
            if attention_masks is None:
                attention_masks = np.ones(input_ids.size(), dtype=np.int64)
            else:
                attention_masks = attention_masks.cpu().numpy()
            data = [input_ids.cpu().numpy(), attention_masks]
            outputs = self.onnxmodel.run(inputs=data)
            for idx, item in enumerate(outputs):
                outputs[idx] = torch.tensor(item, device=input_ids.device)
            return outputs
        elif self.backend == "turbo":
            # torch part
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
            inputs_embeds = try_convert(inputs_embeds)

            # turbo part
            transformer_outputs = self.transformer(
                hidden_states=inputs_embeds,
                attention_mask=attention_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_type=return_type)
            return transformer_outputs

    @staticmethod
    def from_torch(model: TorchDistilBertModel, backend="turbo"):
        """
        :param model: a torch distrilBert Model
        backend： turbo or onnxrt
        move model to gpu before call this function.
        """
        if backend == "turbo":
            transformer = DistrillTransformer.from_torch(model.transformer)
            return DistilBertModel(model.embeddings, transformer, "turbo")
        elif backend == "onnxrt":
            import onnx
            import onnxruntime.backend
            device = model.device
            if 'cuda' in device.type and torch.cuda.is_available():
                use_gpu = True
            else:
                use_gpu = False
            inputs = {
                'input_ids':
                torch.randint(32, [2, 32], dtype=torch.long).to(
                    device),  # list of numerical ids for the tokenised text
                'attention_mask':
                torch.ones([2, 32],
                           dtype=torch.long).to(device),  # dummy list of ones
            }
            onnx_model_path = "/tmp/temp_turbo_onnx.model"
            with open(onnx_model_path, 'wb') as outf:
                torch.onnx.export(
                    model=model,
                    args=(inputs['input_ids'], inputs['attention_mask']
                          ),  # model input (or a tuple for multiple inputs)
                    f=outf,
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': [0, 1],
                        'attention_mask': [0, 1]
                    })
            onnx_model = onnx.load_model(f=onnx_model_path)
            onnx_model = onnxruntime.backend.prepare(
                model=onnx_model,
                device='GPU' if use_gpu else "CPU",
                graph_optimization_level=onnxruntime.GraphOptimizationLevel.
                ORT_ENABLE_ALL)
            return DistilBertModel(onnx_model, None, "onnxrt")
