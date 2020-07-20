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

import turbo_transformers

import unittest
import io
import torch
from transformers.modeling_bert import BertConfig, BertOutput
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_shape_test(batch_size: int, seq_length: int):
    class TestBertOut(unittest.TestCase):
        def init_data(self, use_cuda) -> None:
            test_device = torch.device('cuda:0') if use_cuda else \
                    torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig()
            self.intermediate_size = self.cfg.intermediate_size  # 3072;
            self.hidden_size = self.cfg.hidden_size  # 768
            self.torch_bertout = BertOutput(self.cfg)
            self.torch_bertout.eval()
            if use_cuda:
                self.torch_bertout.to(test_device)

            self.turbo_bertout = turbo_transformers.BertOutput.from_torch(
                self.torch_bertout)

            self.intermediate_output = torch.rand(
                size=(batch_size, seq_length, self.intermediate_size),
                dtype=torch.float32,
                device=test_device)
            self.attention_output = torch.rand(size=(batch_size, seq_length,
                                                     self.hidden_size),
                                               dtype=torch.float32,
                                               device=test_device)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda)
            num_iter = 2
            device = "GPU" if use_cuda else "CPU"

            torch_model = lambda: self.torch_bertout(self.intermediate_output,
                                                     self.attention_output)
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)
            print(f'Bert Output Plain PyTorch({device}) QPS {torch_qps}')

            turbo_model = lambda: self.turbo_bertout(self.intermediate_output,
                                                     self.attention_output)
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)
            print(
                f'Bert Output Plain TurboTransformer({device}) QPS {turbo_qps}'
            )

            # cuda version precision is lower due to tensor-core
            self.assertTrue(
                torch.max(torch.abs(torch_result - turbo_result)) < 1e-2
                if use_cuda else 1e-4)

        def test_bertout(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    TestBertOut.__name__ = f"TestBertOut_BatchSize_{batch_size}_SeqLen_{seq_length}"
    globals()[TestBertOut.__name__] = TestBertOut


for seq_length in (20, 40, 60, 80, 100, 120):
    for batch_size in (1, 2):
        create_shape_test(batch_size=batch_size, seq_length=seq_length)

if __name__ == '__main__':
    unittest.main()
