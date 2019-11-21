import fast_transformers.fast_transformers_cxx as cxx
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

__all__ = [
    'BertEmbeddings', 'BertIntermediate', 'BertOutput', 'BertAttention',
    'BertLayer', 'BertEncoder', 'SequencePool', 'BertModel'
]


def _try_convert(t):
    if isinstance(t, torch.Tensor):
        return convert2ft_tensor(t)
    else:
        return t


def convert2ft_tensor(t):
    return cxx.Tensor.from_dlpack(dlpack.to_dlpack(t))


def _to_param_dict(torch_module: torch.nn.Module):
    return {
        k: convert2ft_tensor(v)
        for k, v in torch_module.named_parameters()
    }


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
                              params['LayerNorm.bias'],
                              bert_embedding.dropout.p)


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
        intermediate_params = _to_param_dict(intermediate)
        return BertIntermediate(intermediate_params['dense.weight'],
                                intermediate_params['dense.bias'])


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
        params = _to_param_dict(output)
        return BertOutput(params["dense.weight"], params["dense.bias"],
                          params["LayerNorm.weight"], params["LayerNorm.bias"])


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
            qkv_weight = torch.cat(
                (params['self.query.weight'], params['self.key.weight'],
                 params['self.value.weight']), 0)
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0)

            att = BertAttention(
                convert2ft_tensor(qkv_weight), convert2ft_tensor(qkv_bias),
                convert2ft_tensor(params['output.dense.weight']),
                convert2ft_tensor(params['output.dense.bias']),
                convert2ft_tensor(params['output.LayerNorm.weight']),
                convert2ft_tensor(params['output.LayerNorm.bias']),
                attention.self.num_attention_heads)

            return att


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
            return_type=ReturnType.FAST_TRANSFORMERS,
            output=attention_output)
        intermediate_output = self.intermediate(
            attention_output,
            return_type=ReturnType.FAST_TRANSFORMERS,
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
                       return_type=ReturnType.FAST_TRANSFORMERS,
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


class SequencePool(cxx.SequencePool):
    def __call__(self,
                 input_tensor: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output_tensor: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        output_tensor = _create_empty_if_none(output_tensor)
        super(SequencePool, self).__call__(input_tensor, output_tensor)
        return convert_returns_as_type(output_tensor, return_type)


class BertModel:
    def __init__(self, embeddings: BertEmbeddings, encoder: BertEncoder,
                 seq_pool: SequencePool):
        self.embeddings = embeddings
        self.encoder = encoder
        self.seq_pool = seq_pool
        self.prepare = cxx.PrepareBertMasks()

    def __call__(self,
                 inputs: AnyTensor,
                 attention_masks: Optional[AnyTensor] = None,
                 token_type_ids: Optional[AnyTensor] = None,
                 position_ids: Optional[AnyTensor] = None,
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
            return_type=ReturnType.FAST_TRANSFORMERS)

        hidden_cache = self.encoder(hidden_states=hidden_cache,
                                    attention_mask=extended_attention_masks,
                                    return_type=ReturnType.FAST_TRANSFORMERS,
                                    output=hidden_cache)

        return self.seq_pool(hidden_cache,
                             return_type=return_type,
                             output_tensor=output)

    @staticmethod
    def from_torch(model: TorchBertModel, pooling_type=None):
        embeddings = BertEmbeddings.from_torch(model.embeddings)
        encoder = BertEncoder.from_torch(model.encoder)
        if pooling_type is None:
            pooling_type = 'First'
        seq_pool = SequencePool(pooling_type)
        return BertModel(embeddings, encoder, seq_pool)
