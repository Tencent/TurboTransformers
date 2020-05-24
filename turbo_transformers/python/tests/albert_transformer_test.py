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
from transformers.modeling_albert import AlbertConfig, AlbertTransformer
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestAlbertTransformer(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')
        if not use_cuda:
            torch.set_num_threads(1)

        torch.set_grad_enabled(False)
        self.cfg = AlbertConfig()

        self.torch_encoder_layer = AlbertTransformer(self.cfg)
        self.torch_encoder_layer.eval()

        if use_cuda:
            self.torch_encoder_layer.to(test_device)

        self.batch_size = 1
        self.seq_length = 40
        self.embedding_size = self.cfg.embedding_size
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length,
                                             self.embedding_size),
                                       dtype=torch.float32,
                                       device=test_device)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length),
                                         dtype=torch.float32,
                                         device=test_device)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        self.turbo_albert_encoder = turbo_transformers.AlbertTransformer.from_torch(self.torch_encoder_layer,
                                                                                    self.cfg)

    def check_torch_and_turbo(self, use_cuda=True):
        self.init_data(use_cuda=use_cuda)
        self.num_iter = 2

        turbo_albert_layer_result = None
        turbo_model = lambda: self.turbo_albert_encoder(
            self.input_tensor,
            self.attention_mask,
            output=turbo_albert_layer_result,
            return_type=turbo_transformers.ReturnType.turbo_transformers)

        turbo_albert_layer_result, turbo_qps, turbo_time_consume = \
            test_helper.run_model(turbo_model, use_cuda, self.num_iter)

        print(f"AlbertTransformer TurboTransform QPS, {turbo_qps}, ",
              f"Time Cost, {turbo_time_consume}")

        turbo_albert_layer_result = self.turbo_albert_encoder(
            self.input_tensor, self.attention_mask)

        torch_model = lambda: self.torch_encoder_layer(
            self.input_tensor, self.attention_mask, [None] * self.cfg.
            num_hidden_layers)

        torch_albert_layer_result, torch_qps, torch_time_consume = \
            test_helper.run_model(torch_model, use_cuda, self.num_iter)

        print(f"AlbertTransformer Torch QPS, {torch_qps}, ",
              f"Time Cost, {torch_time_consume}")

        diff = torch.abs(torch_albert_layer_result[0] - turbo_albert_layer_result)
        print(torch.max(diff))
        self.assertTrue(torch.max(diff) < 2e-2)

    def test_embedding(self):
        self.check_torch_and_turbo(use_cuda=False)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True)


if __name__ == '__main__':
    unittest.main()