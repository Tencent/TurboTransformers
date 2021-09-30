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
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestBertEncoder(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')
        if not use_cuda:
            torch.set_num_threads(1)

        torch.set_grad_enabled(False)
        self.cfg = BertConfig()

        self.torch_encoder_layer = BertEncoder(self.cfg)
        self.torch_encoder_layer.eval()

        if use_cuda:
            self.torch_encoder_layer.to(test_device)

        self.batch_size = 1
        self.seq_length = 40
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length,
                                             self.hidden_size),
                                       dtype=torch.float32,
                                       device=test_device)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length),
                                         dtype=torch.float32,
                                         device=test_device)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        self.turbo_bert_encoder = turbo_transformers.BertEncoder.from_torch(
            self.torch_encoder_layer)

    def check_torch_and_turbo(self, use_cuda=True):
        self.init_data(use_cuda=use_cuda)
        self.num_iter = 2

        turbo_bert_layer_result = None
        turbo_model = lambda: self.turbo_bert_encoder(self.input_tensor,
                                                      self.attention_mask,
                                                      output_attentions=True,
                                                      output_hidden_states=True
                                                      )

        turbo_bert_layer_result, turbo_qps, turbo_time_consume = \
            test_helper.run_model(turbo_model, use_cuda, self.num_iter)

        print(f"BertEncoder TurboTransform QPS, {turbo_qps}, ",
              f"Time Cost, {turbo_time_consume}")

        # turbo_bert_layer_result = self.turbo_bert_encoder(
        #     self.input_tensor,
        #     self.attention_mask,
        #     output_attentions = True,
        #     output_hidden_states = False)

        torch_model = lambda: self.torch_encoder_layer(
            self.input_tensor,
            self.attention_mask, [None] * self.cfg.num_hidden_layers,
            output_attentions=True,
            output_hidden_states=True)

        torch_bert_layer_result, torch_qps, torch_time_consume = \
            test_helper.run_model(torch_model, use_cuda, self.num_iter)

        print(f"BertEncoder Torch QPS, {torch_qps}, ",
              f"Time Cost, {torch_time_consume}")

        diff = torch.abs(torch_bert_layer_result[0] -
                         turbo_bert_layer_result[0])
        self.assertTrue(torch.max(diff) < 1e-2)

        # Note we did not print the last hidden_states, because it is the same as output
        # print(len(torch_bert_layer_result[1]), len(turbo_bert_layer_result[1]))
        # for a, b in zip(torch_bert_layer_result[1],
        #                 turbo_bert_layer_result[1]):
        #     diff = torch.abs(a - b)
        #     self.assertTrue(torch.max(diff) < 1e-2)

        # for a, b in zip(torch_bert_layer_result[2],
        #                 turbo_bert_layer_result[2]):
        #     diff = torch.abs(a - b)
        #     self.assertTrue(torch.max(diff) < 1e-2)

    def test_encoder(self):
        self.check_torch_and_turbo(use_cuda=False)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
