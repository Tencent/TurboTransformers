import fast_transformers.fast_transformers_cxx as cxx
from typing import Union, Optional
import torch
from .return_type import convert_returns_as_type, ReturnType
import torch.utils.dlpack as dlpack

from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate
from transformers.modeling_bert import BertOutput as TorchBertOutput
from transformers.modeling_bert import BertAttention as TorchBertAttention

__all__ = ['BertEmbeddings', 'BertIntermediate', 'BertOutput', 'BertAttention']


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


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self,
                 input_ids: Union[cxx.Tensor, torch.Tensor],
                 position_ids: Union[cxx.Tensor, torch.Tensor],
                 token_type_ids: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None):
        input_ids = _try_convert(input_ids)
        position_ids = _try_convert(position_ids)
        token_type_ids = _try_convert(token_type_ids)

        return convert_returns_as_type(
            super(BertEmbeddings, self).__call__(input_ids, position_ids,
                                                 token_type_ids), return_type)

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
                 input_tensor: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None):
        input_tensor = _try_convert(input_tensor)
        return convert_returns_as_type(
            super(BertIntermediate, self).__call__(input_tensor), return_type)

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = _to_param_dict(intermediate)
        return BertIntermediate(intermediate_params['dense.weight'],
                                intermediate_params['dense.bias'])


class BertOutput(cxx.BertOutput):
    def __call__(self,
                 intermediate_output: Union[cxx.Tensor, torch.Tensor],
                 attention_output: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None):
        intermediate_output = _try_convert(intermediate_output)
        attention_output = _try_convert(attention_output)
        return convert_returns_as_type(
            super(BertOutput, self).__call__(intermediate_output,
                                             attention_output), return_type)

    @staticmethod
    def from_torch(output: TorchBertOutput):
        params = _to_param_dict(output)
        return BertOutput(params["dense.weight"], params["dense.bias"],
                          params["LayerNorm.weight"], params["LayerNorm.bias"])


class BertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: Union[cxx.Tensor, torch.Tensor],
                 attention_mask: Union[cxx.Tensor, torch.Tensor],
                 head_mask: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None):
        input_tensor = _try_convert(input_tensor)
        attention_mask = _try_convert(attention_mask)
        head_mask = _try_convert(head_mask)
        return convert_returns_as_type(
            super(BertAttention, self).__call__(input_tensor, attention_mask,
                                                head_mask), return_type)

    @staticmethod
    def from_torch(attention: TorchBertAttention):
        params = {k: v for k, v in attention.named_parameters()}
        print(attention.self.num_attention_heads)

        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.cat(
                (params['self.query.weight'], params['self.key.weight']), 0)
            qkv_weight = torch.cat((qkv_weight, params['self.value.weight']),
                                   0)
            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias']), 0)
            qkv_bias = torch.cat((qkv_bias, params['self.value.bias']), 0)
        return BertAttention(
            convert2ft_tensor(qkv_weight), convert2ft_tensor(qkv_bias),
            convert2ft_tensor(params['output.dense.weight']),
            convert2ft_tensor(params['output.dense.bias']),
            convert2ft_tensor(params['output.LayerNorm.weight']),
            convert2ft_tensor(params['output.LayerNorm.bias']),
            attention.self.num_attention_heads)
