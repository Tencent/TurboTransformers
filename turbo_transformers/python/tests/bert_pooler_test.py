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

import sys
import torch
import turbo_transformers
from transformers.models.bert.modeling_bert import BertConfig, BertPooler
import numpy
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size):
    class TestBertPooler(unittest.TestCase):
        def init_data(self, use_cuda: bool) -> None:
            self.test_device = torch.device('cuda:0') if use_cuda else \
                    torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig()

            self.torch_pooler = BertPooler(self.cfg)
            if torch.cuda.is_available():
                self.torch_pooler.to(self.test_device)
            self.torch_pooler.eval()

            self.turbo_pooler = turbo_transformers.BertPooler.from_torch(
                self.torch_pooler)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda=use_cuda)
            device = "GPU" if use_cuda else "CPU"

            num_iter = 2
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, 1, hidden_size),
                                      dtype=torch.float32,
                                      device=self.test_device)

            torch_model = lambda: self.torch_pooler(input_tensor)
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)
            print(f"BertPooler \"({batch_size},{hidden_size:03})\" ",
                  f"{device} Torch QPS,  {torch_qps}, time, {torch_time}")

            turbo_model = lambda: self.turbo_pooler(
                input_tensor.reshape((batch_size, hidden_size)))
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)

            print(
                f"BertPooler \"({batch_size}, {hidden_size}\" ",
                f"{device} TurboTransform QPS,  {turbo_qps}, time, {turbo_time}"
            )

            torch_result = torch_result.cpu().numpy()
            turbo_result = turbo_result.cpu().numpy()

            self.assertTrue(
                numpy.allclose(torch_result,
                               turbo_result,
                               rtol=1e-4,
                               atol=1e-3))

            with open("bert_pooler_res.txt", "a") as fh:
                fh.write(
                    f"\"({batch_size},{hidden_size:03})\", {torch_qps}, {torch_qps}\n"
                )

        def test_pooler(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestBertPooler_{batch_size}"] = \
        TestBertPooler


with open("bert_pooler_res.txt", "w") as fh:
    fh.write(", torch, turbo_transformers\n")
    for batch_size in [1, 2, 4, 8, 50, 100]:
        create_test(batch_size)

if __name__ == '__main__':
    unittest.main()
