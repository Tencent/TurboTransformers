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
    NUMPY = 2


def convert_returns_as_type(tensor: cxx.Tensor, rtype: Optional[ReturnType]
                            ) -> Union[cxx.Tensor, torch.Tensor]:
    if rtype is None:
        rtype = ReturnType.TORCH

    if rtype == ReturnType.NUMPY:
        return cxx.tensor2nparrayf(tensor)
    elif rtype == ReturnType.turbo_transformers:
        return tensor
    elif rtype == ReturnType.TORCH:
        return dlpack.from_dlpack(tensor.to_dlpack())
    else:
        raise NotImplementedError()
