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

import unittest
import turbo_transformers
import torch
import numpy as np


def pooling(input, pool_type):
    if pool_type == "First":
        return input[:, 0, :]
    elif pool_type == "Last":
        seq_len = input.shape[1]
        return input[:, seq_len - 1, :]
    elif pool_type == "Mean":
        return np.mean(input, axis=1)
    elif pool_type == "Max":
        return np.max(input, axis=1)
    else:
        raise "{} is not support.".format(pool_type)


def create_test_seq_pool(batch_size: int, seq_length: int, pool_type: str):
    class TestSequencePool(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            hidden_size = 50
            self.input = np.random.random(
                (batch_size, seq_length, hidden_size)).astype("float32")
            self.seq_pool = turbo_transformers.SequencePool(pool_type)

        def check_torch_and_turbo(self, use_cuda):
            test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)
            np_result = pooling(self.input, pool_type)
            turbo_result = self.seq_pool(
                torch.tensor(self.input, device=test_device))
            self.assertTrue(
                np.max(np.abs(np_result - turbo_result.cpu().numpy())) < 1e-3)

        def test_seq_pool(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)



    globals()[f"TestSequencePool{batch_size}_{seq_length:03}_{pool_type}"] = \
        TestSequencePool


for batch_size in [1, 5]:
    for seq_length in [5, 8, 2000]:
        for pool_type in ["Mean", "Max", "First", "Last"]:
            create_test_seq_pool(batch_size, seq_length, pool_type)

if __name__ == '__main__':
    unittest.main()
