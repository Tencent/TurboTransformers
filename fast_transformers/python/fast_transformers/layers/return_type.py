# Copyright 2020 Tencent
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import torch.utils.dlpack as dlpack
try:
    # `fast_transformers_cxxd` is the name on debug mode
    import fast_transformers.fast_transformers_cxxd as cxx
except ImportError:
    import fast_transformers.fast_transformers_cxx as cxx
from typing import Optional, Union
import torch

__all__ = ['ReturnType', 'convert_returns_as_type']


class ReturnType(enum.Enum):
    FAST_TRANSFORMERS = 0
    TORCH = 1
    TENSOR_FLOW = 2


def convert_returns_as_type(tensor: cxx.Tensor, rtype: Optional[ReturnType]
                            ) -> Union[cxx.Tensor, torch.Tensor]:
    if rtype is None:
        rtype = ReturnType.TORCH

    if rtype == ReturnType.FAST_TRANSFORMERS:
        return tensor
    elif rtype == ReturnType.TORCH:
        return dlpack.from_dlpack(tensor.to_dlpack())
    else:
        raise NotImplementedError()
