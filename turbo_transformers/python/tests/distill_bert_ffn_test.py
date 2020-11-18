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
import sys
import torch
import os

from transformers.modeling_distilbert import DistilBertConfig
from transformers.modeling_distilbert import FFN as DistilFFN

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "distrill_ffn.txt"


def create_test(batch_size, input_len):
    class TestDistillFFN(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            self.cfg = DistilBertConfig(attention_probs_dropout_prob=0.0,
                                        hidden_dropout_prob=0.0)

            torch.set_grad_enabled(False)
            self.torch_ffn = DistilFFN(self.cfg)
            self.torch_ffn.eval()
            self.output_layer_norm = torch.nn.LayerNorm(
                normalized_shape=self.cfg.dim, eps=1e-12)
            if use_cuda:
                self.torch_ffn.to(self.test_device)
                self.output_layer_norm.to(self.test_device)

            self.turbo_ffn = turbo_transformers.DistrillFFN.from_torch(
                self.torch_ffn, self.output_layer_norm)
            # (batch_size, input_len, model_dim)
            self.inputs = torch.rand(size=(batch_size, input_len,
                                           self.cfg.dim),
                                     dtype=torch.float32,
                                     device=self.test_device)

            print(self.cfg.activation)

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"

            torch_model = lambda: self.output_layer_norm(
                self.torch_ffn(self.inputs) + self.inputs)
            torch_res, torch_qps, torch_time_consume = \
                test_helper.run_model(torch_model, use_cuda, num_iter)

            print(
                f"DistrillFFN \"({batch_size}, {input_len:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            turbo_res = lambda: self.turbo_ffn(self.inputs,
                                               is_trans_weight=True)
            with turbo_transformers.pref_guard("gpref_test") as perf:
                turbo_res, turbo_qps, turbo_time_consume = \
                    test_helper.run_model(turbo_res, use_cuda, num_iter)

            print(
                f"DistrillFFN \"({batch_size}, {input_len:03})\" ",
                f"{device} Turbo Trans QPS, {turbo_qps}, time, {turbo_time_consume}"
            )

            print(torch.max(torch.abs(torch_res - turbo_res)))
            self.assertTrue(torch.max(torch.abs(torch_res - turbo_res)) < 1e-3)

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{input_len:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_distrill_ffn(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                    turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestDistillFFN{batch_size}_{input_len:3}"] = TestDistillFFN


with open(fname, "w") as fh:
    fh.write(", torch, turbo_trans\n")

for batch_size in [1, 4]:
    for input_len in [10, 20, 30, 40, 50]:
        create_test(batch_size, input_len)

if __name__ == '__main__':
    unittest.main()
