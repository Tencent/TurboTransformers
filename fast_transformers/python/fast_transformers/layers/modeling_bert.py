import fast_transformers.fast_transformers_cxx as cxx
from typing import Union, Optional
import torch
from fast_transformers.torch_utils.converts import convert2ft_tensor
from .return_type import convert_returns_as_type, ReturnType

__all__ = ['BertEmbeddings']


class BertEmbeddings(cxx.BERTEmbedding):
    def __call__(self, input_ids: Union[cxx.Tensor, torch.Tensor],
                 position_ids: Union[cxx.Tensor, torch.Tensor],
                 token_type_ids: Union[cxx.Tensor, torch.Tensor],
                 return_type: Optional[ReturnType] = None) -> cxx.Tensor:
        if isinstance(input_ids, torch.Tensor):
            input_ids = convert2ft_tensor(input_ids)
        if isinstance(position_ids, torch.Tensor):
            position_ids = convert2ft_tensor(position_ids)
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = convert2ft_tensor(token_type_ids)

        return convert_returns_as_type(super(BertEmbeddings, self).__call__(input_ids, position_ids, token_type_ids),
                                       return_type)
