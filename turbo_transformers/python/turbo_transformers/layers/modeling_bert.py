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

from transformers.models.bert.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.models.bert.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.models.bert.modeling_bert import BertOutput as TorchBertOutput
from transformers.models.bert.modeling_bert import BertAttention as TorchBertAttention
from transformers.models.bert.modeling_bert import BertLayer as TorchBertLayer
from transformers.models.bert.modeling_bert import BertEncoder as TorchBertEncoder
from transformers.models.bert.modeling_bert import BertModel as TorchBertModel
from transformers.models.bert.modeling_bert import BertPooler as TorchBertPooler

import enum
import numpy as np

__all__ = [
    'BertEmbeddings', 'BertIntermediate', 'BertOutput', 'BertAttention',
    'BertLayer', 'BertEncoder', 'SequencePool', 'BertModel', 'PoolingType',
    'BertPooler'
]


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self,
                 input_ids: AnyTensor,
                 position_ids: AnyTensor,
                 token_type_ids: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_ids = try_convert(input_ids)
        position_ids = try_convert(position_ids)
        token_type_ids = try_convert(token_type_ids)
        output = create_empty_if_none(output)
        super(BertEmbeddings, self).__call__(input_ids, position_ids,
                                             token_type_ids, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(bert_embedding: TorchBertEmbeddings) -> 'BertEmbeddings':
        params = to_param_dict_convert_tt(bert_embedding)
        return BertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'])

    @staticmethod
    def from_npz(file_name: str, device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertEmbeddings(
            try_convert(f['embeddings.word_embeddings.weight'], device),
            try_convert(f['embeddings.position_embeddings.weight'], device),
            try_convert(f['embeddings.token_type_embeddings.weight'], device),
            try_convert(f['embeddings.LayerNorm.weight'], device),
            try_convert(f['embeddings.LayerNorm.bias'], device))


class BertIntermediate(cxx.BertIntermediate):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(BertIntermediate, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = to_param_dict(intermediate)
        weight = torch.clone(
            torch.t(intermediate_params["dense.weight"]).contiguous())
        return BertIntermediate(
            convert2tt_tensor(weight),
            convert2tt_tensor(intermediate_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertIntermediate(
            try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.intermediate.dense.bias'],
                device))


class BertOutput(cxx.BertOutput):
    def __call__(self,
                 intermediate_output: AnyTensor,
                 attention_output: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        intermediate_output = try_convert(intermediate_output)
        attention_output = try_convert(attention_output)
        output = create_empty_if_none(output)
        super(BertOutput, self).__call__(intermediate_output, attention_output,
                                         output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(output: TorchBertOutput):
        params = to_param_dict(output)
        weight = convert2tt_tensor(
            torch.clone(torch.t(params["dense.weight"]).contiguous()))
        return BertOutput(weight, convert2tt_tensor(params["dense.bias"]),
                          convert2tt_tensor(params["LayerNorm.weight"]),
                          convert2tt_tensor(params["LayerNorm.bias"]))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertOutput(
            try_convert(f[f'encoder.layer.{layer_num}.output.dense.weight'],
                        device),
            try_convert(f[f'encoder.layer.{layer_num}.output.dense.bias'],
                        device),
            try_convert(
                f[f'encoder.layer.{layer_num}.output.LayerNorm.weight'],
                device),
            try_convert(f[f'encoder.layer.{layer_num}.output.LayerNorm.bias'],
                        device))


class BertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 is_trans_weight: Optional[cxx.Tensor] = False):
        """
        implement BertSelfAttention in
        https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py#L183
        self.output_attentions always true
        return (context_layer, attention_probs)
        """
        assert (head_mask is None)
        input_tensor = try_convert(input_tensor)
        attention_mask = try_convert(create_empty_if_none(attention_mask))
        context_layer = cxx.Tensor.create_empty()
        attn_probs = cxx.Tensor.create_empty()
        super(BertAttention,
              self).__call__(input_tensor, attention_mask, context_layer,
                             attn_probs, is_trans_weight)
        outputs = (convert_returns_as_type(context_layer, return_type),
                   convert_returns_as_type(attn_probs,
                                           ReturnType.turbo_transformers)
                   ) if output_attentions else (convert_returns_as_type(
                       context_layer, return_type), )
        return outputs

    @staticmethod
    def from_torch(attention: TorchBertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['self.query.weight'],
                               params['self.key.weight'],
                               params['self.value.weight']),
                              0).contiguous()).contiguous())
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0).contiguous()

            output_weight = torch.clone(
                torch.t(params['output.dense.weight']).contiguous())
            att = BertAttention(
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                convert2tt_tensor(output_weight),
                convert2tt_tensor(params['output.dense.bias']),
                convert2tt_tensor(params['output.LayerNorm.weight']),
                convert2tt_tensor(params['output.LayerNorm.bias']),
                attention.self.num_attention_heads)

            return att

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertAttention(
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.weight'],
                        device),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.bias'],
                        device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.bias'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.weight'],
                device),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.bias'],
                device), num_attention_heads)


class BertLayer:
    def __init__(self, attention: BertAttention,
                 intermediate: BertIntermediate, output: BertOutput):
        self.attention = attention
        self.intermediate = intermediate
        self.output = output

    def __call__(self,
                 hidden_states: AnyTensor,
                 attention_mask: Optional[AnyTensor] = None,
                 head_mask: Optional[AnyTensor] = None,
                 output_attentions=False,
                 return_type: Optional[ReturnType] = None):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
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
        return BertLayer(BertAttention.from_torch(layer.attention),
                         BertIntermediate.from_torch(layer.intermediate),
                         BertOutput.from_torch(layer.output))

    @staticmethod
    def from_npz(file_name: str,
                 layer_num: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertLayer(
            BertAttention.from_npz(file_name, layer_num, num_attention_heads,
                                   device),
            BertIntermediate.from_npz(file_name, layer_num, device),
            BertOutput.from_npz(file_name, layer_num, device))


class BertEncoder:
    def __init__(self, layer: Sequence[BertLayer]):
        self.layer = layer

    def __call__(self,
                 hidden_states: AnyTensor,
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
                              attention_mask=attention_mask,
                              output_attentions=output_attentions,
                              return_type=ReturnType.turbo_transformers)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states,
                                            ReturnType.turbo_transformers), )

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
            BertLayer.from_torch(bert_layer) for bert_layer in encoder.layer
        ]
        return BertEncoder(layer)

    @staticmethod
    def from_npz(file_name: str,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 device: Optional[torch.device] = None):
        layer = []
        for i in range(num_hidden_layers):
            layer.append(
                BertLayer.from_npz(file_name, i, num_attention_heads, device))
        return BertEncoder(layer)


class SequencePool(cxx.SequencePool):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output_tensor: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output_tensor = create_empty_if_none(output_tensor)
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


class BertPooler(cxx.BertPooler):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(BertPooler, self).__call__(input_tensor, output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(pooler: TorchBertPooler):
        pooler_params = to_param_dict(pooler)
        weight = torch.clone(
            torch.t(pooler_params['dense.weight']).contiguous())
        return BertPooler(convert2tt_tensor(weight),
                          convert2tt_tensor(pooler_params['dense.bias']))

    @staticmethod
    def from_npz(file_name: str, device: Optional[torch.device] = None):
        f = np.load(file_name)
        return BertPooler(try_convert(f['pooler.dense.weight'], device),
                          try_convert(f['pooler.dense.bias'], device))


class BertModelNoPooler:
    def __init__(self, embeddings: BertEmbeddings, encoder: BertEncoder):
        self.embeddings = embeddings
        self.encoder = encoder
        self.prepare = cxx.PrepareBertMasks()

    def __call__(
            self,
            inputs: AnyTensor,
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
        attention_masks = try_convert(create_empty_if_none(attention_masks))
        token_type_ids = try_convert(create_empty_if_none(token_type_ids))
        position_ids = try_convert(create_empty_if_none(position_ids))
        inputs = try_convert(inputs)
        extended_attention_masks = cxx.Tensor.create_empty()

        self.prepare(inputs, attention_masks, token_type_ids, position_ids,
                     extended_attention_masks)

        hidden_cache = self.embeddings(
            inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            return_type=ReturnType.turbo_transformers)

        encoder_outputs = self.encoder(
            hidden_states=hidden_cache,
            attention_mask=extended_attention_masks,
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
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        return BertModelNoPooler(embeddings, encoder)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModelNoPooler.from_torch(torch_model, device)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        embeddings = BertEmbeddings.from_npz(file_name, device)
        encoder = BertEncoder.from_npz(file_name, config.num_hidden_layers,
                                       config.num_attention_heads, device)
        return BertModelNoPooler(embeddings, encoder)


class BertModel:
    # @params:
    # pooler is used for turbo backend only
    # config is used for memory optizations
    def __init__(self, model, pooler=None, backend="onnxrt", config=None):
        # TODO type of bertmodel_nopooler is (onnx and torch)
        self.backend = backend
        if backend == "onnxrt":
            self.onnxmodel = model
        elif backend == "turbo":
            self.config = config
            self.bertmodel_nopooler = model
            self.pooler = pooler
            self.backend = "turbo"

    def __call__(self,
                 inputs: AnyTensor,
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
        if self.backend == "turbo":
            encoder_outputs = self.bertmodel_nopooler(
                inputs,
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
        elif self.backend == "onnxrt":
            if attention_masks is None:
                attention_masks = np.ones(inputs.size(), dtype=np.int64)
            else:
                attention_masks = attention_masks.cpu().numpy()
            if token_type_ids is None:
                token_type_ids = np.zeros(inputs.size(), dtype=np.int64)
            else:
                token_type_ids = token_type_ids.cpu().numpy()
            data = [inputs.cpu().numpy(), attention_masks, token_type_ids]
            outputs = self.onnxmodel.run(inputs=data)
            for idx, item in enumerate(outputs):
                outputs[idx] = torch.tensor(item, device=inputs.device)
            return outputs

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None,
                   backend: Optional[str] = None,
                   use_memory_opt=False):
        """
        Args:
            model : a PyTorch Bert Model
            device : cpu or GPU
            backend : a string to indicates kernel provides
            Four options. [onnxrt-cpu, onnxrt-gpu, turbo-cpu, turbo-gpu]
            use_memory_opt [bool] whether or not use memory opt for variable length inputs.
        """
        use_gpu = False
        if device is None:
            device = model.device
        # we may need to move to GPU explicitly
        if 'cuda' in device.type and torch.cuda.is_available():
            model.to(device)
            if backend is None:
                backend = "turbo"  # On GPU turbo is faster
            use_gpu = True
        else:
            if backend is None:
                backend = "onnxrt"  # On CPU onnxrt is faster

        if backend == "turbo":
            embeddings = BertEmbeddings.from_torch(model.embeddings)
            encoder = BertEncoder.from_torch(model.encoder)
            bertmodel_nopooler = BertModelNoPooler(embeddings, encoder)
            pooler = BertPooler.from_torch(model.pooler)
            return BertModel(bertmodel_nopooler, pooler, "turbo", model.config)
        elif backend == "onnxrt":
            import onnx
            import onnxruntime
            import onnxruntime.backend
            inputs = {
                'input_ids':
                torch.randint(32, [2, 32], dtype=torch.long).to(
                    device),  # list of numerical ids for the tokenised text
                'attention_mask':
                torch.ones([2, 32],
                           dtype=torch.long).to(device),  # dummy list of ones
                'token_type_ids':
                torch.ones([2, 32],
                           dtype=torch.long).to(device),  # dummy list of ones
            }
            onnx_model_path = "/tmp/temp_turbo_onnx.model"
            with open(onnx_model_path, 'wb') as outf:
                torch.onnx.export(
                    model=model,
                    args=(inputs['input_ids'], inputs['attention_mask'],
                          inputs['token_type_ids']
                          ),  # model input (or a tuple for multiple inputs)
                    f=outf,
                    input_names=[
                        'input_ids', 'attention_mask', 'token_type_ids'
                    ],
                    opset_version=11,  # the ONNX version to export the model to
                    do_constant_folding=
                    True,  # whether to execute constant folding for optimization
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': [0, 1],
                        'attention_mask': [0, 1],
                        'token_type_ids': [0, 1]
                    })
            # num_threads = "8"
            # os.environ['OMP_NUM_THREADS'] = str(num_threads)
            # os.environ['MKL_NUM_THREADS'] = str(num_threads)
            onnx_model = onnx.load_model(f=onnx_model_path)
            onnx_model = onnxruntime.backend.prepare(
                model=onnx_model,
                device='GPU' if use_gpu else "CPU",
                graph_optimization_level=onnxruntime.GraphOptimizationLevel.
                ORT_ENABLE_ALL)
            return BertModel(onnx_model, None, "onnxrt")

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None,
                        backend: Optional[str] = None):
        torch_model = TorchBertModel.from_pretrained(model_id_or_path)
        model = BertModel.from_torch(torch_model, device, backend,
                                     torch_model.config)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        model = BertModelNoPooler.from_npz(file_name, config, device)
        pooler = BertPooler.from_npz(file_name, device)
        return BertModel(model, pooler, backend="turbo")
