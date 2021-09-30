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

from transformers.models.distilbert.modeling_distilbert import DistilBertConfig
from transformers.models.distilbert.modeling_distilbert import DistilBertModel

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "distrill_tbert.txt"


def create_test(batch_size, input_len):
    class TestDistillBertModel(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            self.cfg = DistilBertConfig(attention_probs_dropout_prob=0.0,
                                        hidden_dropout_prob=0.0)

            torch.set_grad_enabled(False)
            self.torch_model = DistilBertModel(self.cfg)
            self.torch_model.eval()
            if use_cuda:
                self.torch_model.to(self.test_device)

            self.turbo_transformer = turbo_transformers.DistilBertModel.from_torch(
                self.torch_model)
            # (batch_size, input_len, model_dim)
            self.inputs = torch.randint(low=0,
                                        high=self.cfg.vocab_size - 1,
                                        size=(batch_size, input_len),
                                        dtype=torch.long,
                                        device=self.test_device)
            self.attention_mask = torch.ones((batch_size, input_len),
                                             dtype=torch.long,
                                             device=self.test_device)
            self.head_mask = [None] * self.cfg.num_hidden_layers

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"

            torch_model = lambda: self.torch_model(self.inputs, self.
                                                   attention_mask)
            torch_res, torch_qps, torch_time_consume = \
                test_helper.run_model(torch_model, use_cuda, num_iter)

            print(
                f"DistillBertModel \"({batch_size}, {input_len:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            turbo_res = lambda: self.turbo_transformer(
                self.inputs, self.attention_mask, head_mask=self.head_mask)
            with turbo_transformers.pref_guard("gpref_test") as perf:
                turbo_res, turbo_qps, turbo_time_consume = \
                    test_helper.run_model(turbo_res, use_cuda, num_iter)

            print(
                f"DistillBertModel \"({batch_size}, {input_len:03})\" ",
                f"{device} Turbo QPS, {turbo_qps}, time, {turbo_time_consume}")

            self.assertTrue(
                torch.max(torch.abs(torch_res[0] - turbo_res[0])) < 1e-2
                if use_cuda else 1e-3)

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{input_len:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_distrill_bert_model(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                    turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestDistillTBertModel{batch_size}_{input_len:3}"] = TestDistillBertModel


with open(fname, "w") as fh:
    fh.write(", torch, turbo_trans\n")

for batch_size in [4]:
    for input_len in [10]:
        create_test(batch_size, input_len)

if __name__ == '__main__':
    unittest.main()
