import fast_transformers.fast_transformers_cxx as cxx
from typing import Union, Optional
import torch
from .return_type import convert_returns_as_type, ReturnType
import torch.utils.dlpack as dlpack
import fast_transformers.fast_transformers_cxx as fast_transformers

from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate

__all__ = ['BertEmbeddings', 'BertIntermediate']


def _try_convert(t):
    if isinstance(t, torch.Tensor):
        return convert2ft_tensor(t)
    else:
        return t


def convert2ft_tensor(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self, input_ids: Union[cxx.Tensor, torch.Tensor],
                 position_ids: Union[cxx.Tensor, torch.Tensor],
                 token_type_ids: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None):
        input_ids = _try_convert(input_ids)
        position_ids = _try_convert(position_ids)
        token_type_ids = _try_convert(token_type_ids)

        return convert_returns_as_type(super(BertEmbeddings, self).__call__(input_ids, position_ids, token_type_ids),
                                       return_type)

    @staticmethod
    def from_torch(bert_embedding: TorchBertEmbeddings) -> 'BertEmbeddings':
        params = {k: convert2ft_tensor(v) for k, v in
                  bert_embedding.named_parameters()}

        return BertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'], bert_embedding.dropout.p)


class BertIntermediate(cxx.BertIntermediate):
    def __call__(self, input_tensor: Union[cxx.Tensor, torch.Tensor], return_type: Optional[ReturnType] = None):
        input_tensor = _try_convert(input_tensor)
        return convert_returns_as_type(super(BertIntermediate, self).__call__(input_tensor), return_type)

    @staticmethod
    def from_torch(intermediate: TorchBertIntermediate):
        intermediate_params = {k: convert2ft_tensor(v) for k, v in intermediate.named_parameters()}
        return BertIntermediate(intermediate_params['dense.weight'], intermediate_params['dense.bias'])
