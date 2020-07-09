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
from transformers.modeling_albert import AlbertConfig, AlbertModel
import numpy
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size, seq_length):
    class TestAlbertModel(unittest.TestCase):
        def init_data(self, use_cuda: bool) -> None:
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.cfg = AlbertConfig()

            self.torch_model = AlbertModel(self.cfg)
            if torch.cuda.is_available():
                self.torch_model.to(self.test_device)
            self.torch_model.eval()
            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.randint(low=0,
                                              high=self.cfg.vocab_size - 1,
                                              size=(batch_size, seq_length),
                                              device=self.test_device)

            self.turbo_model = turbo_transformers.AlbertModel.from_torch(
                self.torch_model)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda=use_cuda)
            device = "GPU" if use_cuda else "CPU"
            num_iter = 1
            turbo_model = lambda: self.turbo_model(
                self.input_tensor, attention_mask=None, head_mask=None)
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)

            print(
                f"AlbertLayer \"({batch_size},{seq_length:03})\" ",
                f"{device} TurboTransform QPS,  {turbo_qps}, time, {turbo_time}"
            )

            torch_model = lambda: self.torch_model(input_ids=self.input_tensor,
                                                   attention_mask=None,
                                                   head_mask=None)
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)

            print(f"AlbertModel \"({batch_size},{seq_length:03})\" ",
                  f"{device} Torch QPS,  {torch_qps}, time, {torch_time}")

            # print(turbo_result[-1])
            # print(turbo_result, torch_result[0])
            # TODO(jiaruifang) Error is too high. Does tensor core introduce more differences?
            tolerate_error = 1e-2
            self.assertTrue(
                torch.max(torch.abs(torch_result[0] -
                                    turbo_result[0])) < tolerate_error)

            with open("albert_model_res.txt", "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {torch_qps}\n"
                )

        def test_layer(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestAlbertModel{batch_size}_{seq_length:03}"] = \
        TestAlbertModel


with open("albert_model_res.txt", "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 60, 120]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
