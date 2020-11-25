import enum
import torch
import torch.utils.dlpack as dlpack
try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from typing import Optional, Union

__all__ = ['ReturnType', 'convert_returns_as_type']


class ReturnType(enum.Enum):
    turbo_transformers = 0
    TORCH = 1
    TENSOR_FLOW = 2


def convert_returns_as_type(tensor: cxx.Tensor, rtype: Optional[ReturnType]
                            ) -> Union[cxx.Tensor, torch.Tensor]:
    if rtype is None:
        rtype = ReturnType.TORCH

    if rtype == ReturnType.turbo_transformers:
        return tensor
    elif rtype == ReturnType.TORCH:
        return dlpack.from_dlpack(tensor.to_dlpack())
    else:
        raise NotImplementedError()
