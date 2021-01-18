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

import numpy as np
import torch
import torch.utils.dlpack as dlpack
import unittest
import turbo_transformers


class TestNumpy2Cxx(unittest.TestCase):
    def test(self):
        np_data = np.random.rand(2, 3, 1)
        print(np_data)
        t = turbo_transformers.nparray2tensorf(np_data)

        b = turbo_transformers.tensor2nparrayf(t)
        print(b)


if __name__ == '__main__':
    unittest.main()
