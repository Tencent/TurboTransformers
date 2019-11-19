import torch.utils.dlpack as dlpack
import fast_transformers.fast_transformers_cxx as fast_transformers

__all__ = ['convert2ft_tensor']


def convert2ft_tensor(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))
