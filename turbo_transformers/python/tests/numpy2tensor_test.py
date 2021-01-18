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
    def test_cpu2gpu(self):
        np_data = np.random.rand(2, 3, 1)
        print(np_data)
        t = turbo_transformers.nparray2tensorf(np_data, "GPU")
        np_data_res = turbo_transformers.tensor2nparrayf(t, "GPU")
        assert (np.max(np.abs(np_data_res - np_data) < 1e-6))

    def test_cpu2cpu(self):
        np_data = np.random.rand(2, 3, 1)
        print(np_data.shape)
        t = turbo_transformers.nparray2tensorf(np_data, "CPU")

        np_data_res = turbo_transformers.tensor2nparrayf(t, "CPU")
        assert (np.max(np.abs(np_data_res - np_data) < 1e-6))
        print(np_data_res.shape)


if __name__ == '__main__':
    unittest.main()
