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
from transformers.modeling_albert import AlbertEmbeddings as TorchAlbertEmbeddings
from transformers.modeling_albert import AlbertTransformer as TorchAlbertTransformer
from transformers.modeling_albert import AlbertAttention as TorchAlbertAttention
from transformers.modeling_albert import AlbertLayer as TorchAlbertLayer
from transformers.modeling_albert import AlbertLayerGroup as TorchAlbertLayerGroup
import torch
import enum


__all__ = ["AlbertEmbeddings", "AlbertTransformer", "AlbertAttention","AlbertLayerGroup"
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


# The implement of AlbertEmbeddings is totally equivalent with AlbertEmbedding, So nothing new to do
class AlbertEmbeddings(cxx.BERTEmbedding):
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
        super(AlbertEmbeddings, self).__call__(input_ids, position_ids,
                                             token_type_ids, output)
        return convert_returns_as_type(output, return_type)
    @staticmethod
    def from_torch(albert_embedding: TorchAlbertEmbeddings) -> 'AlbertEmbeddings':
        params = _to_param_dict(albert_embedding)
        return AlbertEmbeddings(params['word_embeddings.weight'],
                              params['position_embeddings.weight'],
                              params['token_type_embeddings.weight'],
                              params['LayerNorm.weight'],
                              params['LayerNorm.bias'])


#AlbertAttention seems like a combination of AlbertAttention and AlbertOutput , So just need a small modification.
class AlbertAttention(cxx.BertAttention):
    def __call__(self,
                 input_tensor: AnyTensor,
                 attention_mask: AnyTensor,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        input_tensor = _try_convert(input_tensor)
        attention_mask = _try_convert(attention_mask)
        output = _create_empty_if_none(output)
        super(AlbertAttention, self).__call__(input_tensor, attention_mask,
                                            output)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_torch(attention: TorchAlbertAttention):
        params = {k: v for k, v in attention.named_parameters()}

        with torch.no_grad():
            # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((params['query.weight'],
                               params['key.weight'],
                               params['value.weight']), 0)))
            qkv_bias = torch.cat(
                (params['query.bias'], params['key.bias'],
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


