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
from turbo_transformers.layers.modeling_bert import try_convert
from turbo_transformers.layers.modeling_bert import convert2tt_tensor


class TestNumpy2Cxx(unittest.TestCase):
    def check_try_convert(self, use_cuda):
        test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')
        if not use_cuda:
            torch.set_num_threads(1)

        a = np.random.rand(4, 3)
        tensor = try_convert(a, test_device)
        a_ref = turbo_transformers.tensor2nparrayf(tensor)
        assert (np.max(np.abs(a - a_ref)) < 1e-6)

    def check_float_convert(self, dev_type):
        np_data = np.random.rand(2, 3, 1)
        t = turbo_transformers.nparray2tensor(np_data, dev_type)
        np_data_res = turbo_transformers.tensor2nparrayf(t)
        assert (np.max(np.abs(np_data_res.reshape(-1) - np_data.reshape(-1))) <
                1e-6)
        assert ((np_data[1][2][0] - np_data_res[1][2][0]) < 1e-6)

    def check_long_convert(self, dev_type):
        np_data = np.random.randint(low=0,
                                    high=10,
                                    size=(2, 3, 1),
                                    dtype=np.int64)
        t = turbo_transformers.nparray2tensor(np_data, dev_type)
        np_data_res = turbo_transformers.tensor2nparrayl(t)
        assert (np.max(np.abs(np_data_res.reshape(-1) - np_data.reshape(-1))) <
                1e-6)
        assert ((np_data[1][2][0] - np_data_res[1][2][0]) < 1e-6)

    def test(self):
        self.check_float_convert("CPU")
        self.check_long_convert("CPU")
        self.check_try_convert(False)
        if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
            self.check_float_convert("GPU")
            self.check_long_convert("GPU")
            self.check_try_convert(True)


if __name__ == '__main__':
    unittest.main()
