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


def create_test(batch_size, key_seq_len, query_seq_len, attn_type,
                pre_layernorm, post_add):
    class TestMultiHeadedAttention(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.head_count = 12
            self.model_dim = 768  #self.model_dim should % self.head_count = 0

            onmt_multi_headed_attention = MultiHeadedAttention(
                self.head_count, self.model_dim)
            onmt_multi_headed_attention.eval()
            if use_cuda:
                onmt_multi_headed_attention.to(self.test_device)

            torch_layernorm = torch.nn.LayerNorm(self.model_dim, eps=1e-6)

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
                onmt_multi_headed_attention, torch_layernorm)
            return onmt_multi_headed_attention, torch_layernorm, turbo_multi_headed_attention, Q, K, V

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            if use_cuda:
                return
            onmt_multi_headed_attention, torch_layernorm, turbo_multi_headed_attention, Q, K, V = \
                self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"
            info = f"\"({pre_layernorm}, {attn_type}, {batch_size}, {key_seq_len:03}, {query_seq_len:03})\""
            attention_mask = torch.zeros(
                (batch_size, 1, key_seq_len if (attn_type == "context") else
                 query_seq_len),  #TODO mask shape is diff for context and self
                dtype=torch.bool,
                device=self.test_device)

            onmt_model = lambda: onmt_multi_headed_attention(
                K,
                V,
                torch.clone(torch_layernorm(Q)) if pre_layernorm else Q,
                attention_mask,
                layer_cache={
                    "memory_keys": None,
                    "memory_values": None,
                    "self_values": None,
                    "self_keys": None
                },
                attn_type=attn_type)

            with turbo_transformers.gperf_guard("gpref_output") as gperf:
                onmt_multi_headed_attention_result, torch_qps, torch_time_consume = \
                    test_helper.run_model(onmt_model, use_cuda, num_iter) # return output, attns

            if post_add:
                onmt_res = onmt_multi_headed_attention_result[0] + Q
            else:
                onmt_res = onmt_multi_headed_attention_result[0]
            print(
                f"ONMT Multi Headed Attention {info} ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            attention_mask = torch.ones(
                (batch_size, 1, key_seq_len if (attn_type == "context") else
                 query_seq_len),  #TODO mask shape is diff for context and self
                dtype=torch.float32,
                device=self.test_device)

            turbo_attention_mask = (1.0 - attention_mask) * -1e18
            turob_model = lambda: turbo_multi_headed_attention(
                K,
                V,
                Q,
                turbo_attention_mask,
                layer_cache=None,
                attn_type=attn_type,
                pre_layernorm=pre_layernorm,
                post_add=post_add)
            turbo_result, turbo_qps, turbo_time_consume = \
                test_helper.run_model(turob_model, use_cuda,
                                      num_iter)
            print(
                f"Turbo Multi Headed Attention {info}",
                f" {device} Turbo QPS, {turbo_qps}, time, {turbo_time_consume}"
            )

            # print(onmt_res)
            # print(turbo_result)

            self.assertTrue(
                torch.max(torch.abs(onmt_res - turbo_result[0])) < (
                    1e-3 if use_cuda else 1e-4))
            with open(fname, "a") as fh:
                fh.write(f", {info} {torch_qps}, {turbo_qps}\n")

        def test_multi_headed_attention(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestMultiHeadedAttention{batch_size}_{key_seq_len:3}_{query_seq_len:3}_{attn_type}_{pre_layernorm}"] = TestMultiHeadedAttention


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")

# for post_add in [False, True]:
#     for pre_layernorm in [False, True]:
#         for attn_type in ["self", "context"]:
#             for batch_size in [1, 2]:
#                 for key_seq_len in [10, 16, 20, 30]:
#                     for query_seq_len in [10, 16, 20, 30]:
for post_add in [False]:
    for pre_layernorm in [False]:
        for attn_type in ["self"]:
            for batch_size in [2]:
                for key_seq_len in [10]:
                    for query_seq_len in [30]:
                        create_test(batch_size, key_seq_len, query_seq_len,
                                    attn_type, pre_layernorm, post_add)

if __name__ == '__main__':
    unittest.main()
