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

from onmt.modules.multi_headed_attn import MultiHeadedAttention

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_multi_headed_attention.txt"


def create_test(batch_size, key_seq_len, query_seq_len, attn_type):
    class TestMultiHeadedAttention(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.attn_type = attn_type
            self.head_count = 12
            self.model_dim = 768  #self.model_dim should % self.head_count = 0

            onmt_multi_headed_attention = MultiHeadedAttention(
                self.head_count, self.model_dim)
            onmt_multi_headed_attention.eval()
            if use_cuda:
                onmt_multi_headed_attention.to(self.test_device)

            K = torch.rand(
                size=(
                    batch_size,
                    key_seq_len,  #from_seq
                    self.model_dim),
                dtype=torch.float32,
                device=self.test_device)
            V = torch.rand(size=(batch_size, key_seq_len, self.model_dim),
                           dtype=torch.float32,
                           device=self.test_device)
            Q = torch.rand(
                size=(
                    batch_size,
                    query_seq_len,  #to_seq
                    self.model_dim),
                dtype=torch.float32,
                device=self.test_device)
            turbo_multi_headed_attention = turbo_transformers.MultiHeadedAttention.from_onmt(
                onmt_multi_headed_attention)
            return onmt_multi_headed_attention, turbo_multi_headed_attention, Q, K, V

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            onmt_multi_headed_attention, turbo_multi_headed_attention, Q, K, V = \
                self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"

            attention_mask = torch.zeros((batch_size, 1, query_seq_len),
                                         dtype=torch.bool,
                                         device=self.test_device)

            onmt_model = lambda: onmt_multi_headed_attention(
                K,
                V,
                Q,
                attention_mask,
                layer_cache={
                    "self_keys": None,
                    "self_values": None
                },
                attn_type=self.attn_type)
            onmt_multi_headed_attention_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter) # return output, attns
            print(
                f"ONMT Multi Headed Attention \"({self.attn_type}, {batch_size},{key_seq_len:03},{query_seq_len:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            attention_mask = torch.ones(
                (batch_size, self.head_count, query_seq_len, key_seq_len),
                dtype=torch.float32,
                device=self.test_device)

            turbo_attention_mask = (1.0 - attention_mask) * -1e18
            turob_model = lambda: turbo_multi_headed_attention(
                K,
                V,
                Q,
                turbo_attention_mask,
                layer_cache=None,
                attn_type=self.attn_type)
            turbo_self_attention_result, turbo_qps, turbo_time_consume = \
                test_helper.run_model(turob_model, use_cuda,
                                      num_iter)
            print(
                f"Turbo Multi Headed Attention  \"({self.attn_type}, {batch_size},{key_seq_len:03},{query_seq_len:03})\" ",
                f" {device} Turbo QPS, {turbo_qps}, time, {turbo_time_consume}"
            )

            self.assertTrue(
                torch.max(
                    torch.abs(onmt_multi_headed_attention_result[0] -
                              turbo_self_attention_result)) < (
                                  1e-3 if use_cuda else 1e-4))
            with open(fname, "a") as fh:
                fh.write(
                    f"\"({self.attn_type},{batch_size},{key_seq_len:03},{query_seq_len:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_multi_headed_attention(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestBertAtt{batch_size}_{key_seq_len:3}_{query_seq_len:3}"] = TestMultiHeadedAttention


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")

for attn_type in {"self"}:
    for batch_size in [1, 2]:
        for key_seq_len in [10, 16, 20, 30]:
            for query_seq_len in [10, 16, 20, 30]:
                create_test(batch_size, key_seq_len, query_seq_len, attn_type)

if __name__ == '__main__':
    unittest.main()
