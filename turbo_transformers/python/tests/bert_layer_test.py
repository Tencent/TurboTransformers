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

import torch
from transformers.models.bert.modeling_bert import BertConfig, BertLayer
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_bertlayer.txt"


def create_test(batch_size, seq_length):
    class TestBertLayer(unittest.TestCase):
        def init_data(self, use_cuda: bool) -> None:
            test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)

            self.torch_bert_layer = BertLayer(self.cfg)
            self.torch_bert_layer.eval()
            if use_cuda:
                self.torch_bert_layer.to(test_device)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32,
                                           device=test_device)

            self.attention_mask = torch.ones((batch_size, seq_length),
                                             dtype=torch.float32,
                                             device=test_device)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

            self.turbo_bert_layer = turbo_transformers.BertLayer.from_torch(
                self.torch_bert_layer)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda)
            num_iter = 2
            device = "GPU" if use_cuda else "CPU"
            torch_model = lambda: self.torch_bert_layer(
                self.input_tensor, self.attention_mask, output_attentions=True)
            torch_bert_layer_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)
            print(f"BertLayer \"({batch_size},{seq_length:03})\" ",
                  f"{device} Torch QPS,  {torch_qps}, time, {torch_time}")

            turbo_model = lambda: self.turbo_bert_layer(
                self.input_tensor, self.attention_mask, output_attentions=True)
            turbo_bert_layer_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)
            print(
                f"BertLayer \"({batch_size},{seq_length:03})\"  ",
                f"{device} TurboTransform QPS, {turbo_qps}, time, {turbo_time}"
            )

            # Tensor core will introduce more errors
            tolerate_error = 1e-2 if use_cuda else 1e-3
            self.assertTrue(
                torch.max(
                    torch.abs(torch_bert_layer_result[0] -
                              turbo_bert_layer_result[0])) < tolerate_error)

            # self.assertTrue(
            #     torch.max(
            #         torch.abs(torch_bert_layer_result[1] -
            #                   turbo_bert_layer_result[1])) < tolerate_error)

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_bert_layer(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestBertLayer{batch_size}_{seq_length:03}"] = TestBertLayer


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 20]:
    for seq_length in [10, 20]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
