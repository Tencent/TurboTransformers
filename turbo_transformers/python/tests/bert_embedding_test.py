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
from transformers.modeling_bert import BertEmbeddings, BertConfig
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test_bert_emb(batch_size: int, seq_length: int):
    class TestBertEmbedding(unittest.TestCase):
        def init_data(self, use_cuda: bool):
            test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')

            torch.set_grad_enabled(False)
            cfg = BertConfig()
            self.torch_embedding = BertEmbeddings(cfg)

            self.torch_embedding.eval()

            if use_cuda:
                self.torch_embedding.to(test_device)

            self.turbo_embedding = turbo_transformers.BertEmbeddings.from_torch(
                self.torch_embedding)

            input_ids = torch.randint(low=0,
                                      high=cfg.vocab_size - 1,
                                      size=(batch_size, seq_length),
                                      dtype=torch.long,
                                      device=test_device)
            position_ids = torch.arange(seq_length,
                                        dtype=torch.long,
                                        device=input_ids.device)

            position_ids = position_ids.repeat(batch_size, 1)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

            return input_ids, position_ids, token_type_ids

        def check_torch_and_turbo(self, use_cuda):
            input_ids, position_ids, token_type_ids = self.init_data(use_cuda)

            device = "GPU" if use_cuda else "CPU"
            num_iter = 100
            torch_model = lambda: self.torch_embedding(
                input_ids, token_type_ids, position_ids)
            torch_result, torch_qps, torch_time = test_helper.run_model(
                torch_model, use_cuda, num_iter)
            print(f"BertEmbeddings \"({batch_size},{seq_length:03})\" ",
                  f"{device} Torch QPS,  {torch_qps}, time, {torch_time}")

            turbo_model = lambda: self.turbo_embedding(input_ids, position_ids,
                                                       token_type_ids)
            turbo_result, turbo_qps, turbo_time = test_helper.run_model(
                turbo_model, use_cuda, num_iter)
            print(f"BertEmbeddings \"({batch_size},{seq_length:03})\" ",
                  f"{device} Turbo QPS,  {turbo_qps}, time, {turbo_time}")

            self.assertTrue(
                torch.max(torch.abs(torch_result - turbo_result)) < 1e-5)

        def test_embedding(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestBertEmbedding{batch_size}_{seq_length:03}"] = \
        TestBertEmbedding


for batch_size in [1, 2]:
    for seq_length in [10, 20, 40, 80, 100, 120]:
        create_test_bert_emb(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
