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

import torch
import torch.utils.dlpack as dlpack
import unittest
from easy_transformers.layers.modeling_bert import convert2ft_tensor
import easy_transformers


class TestDLPack(unittest.TestCase):
    def test_dlpack(self):
        if not torch.cuda.is_available(
        ) or not easy_transformers.config.is_with_cuda():
            torch.set_num_threads(1)
            self.test_device = torch.device('cpu')
            self.device = "CPU"
        else:
            self.test_device = torch.device('cuda:0')
            self.device = "GPU"
        a = torch.rand(size=(4, 3),
                       dtype=torch.float32,
                       device=self.test_device)
        tensor = convert2ft_tensor(a)
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())

        self.assertTrue(a.equal(b))
        self.assertTrue(b.cpu().equal(a.cpu()))


if __name__ == '__main__':
    unittest.main()
