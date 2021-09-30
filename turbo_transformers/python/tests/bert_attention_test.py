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
from transformers.models.bert.modeling_bert import BertConfig, BertAttention
import os

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_attention.txt"


def create_test(batch_size, seq_length):
    class TestBertAttention(unittest.TestCase):
        def init_data(self, use_cuda):
            test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)
            self.cfg.output_attentions = True
            torch_attention = BertAttention(self.cfg)
            torch_attention.eval()
            if use_cuda:
                torch_attention.to(test_device)

            # Get FT Attention
            turbo_attention = turbo_transformers.BertAttention.from_torch(
                torch_attention)

            turbo_decoder_attention = turbo_transformers.MultiHeadedAttention.from_torch(
                torch_attention, is_trans_weight=False)

            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32,
                                      device=test_device)
            attention_mask = torch.ones((batch_size, seq_length),
                                        dtype=torch.float32,
                                        device=test_device)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            return torch_attention, turbo_attention, turbo_decoder_attention, input_tensor, attention_mask

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            torch_attention, turbo_attention, turbo_decoder_attention, input_tensor, attention_mask = \
                self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"
            torch_model = lambda: torch_attention(input_tensor,
                                                  attention_mask,
                                                  output_attentions=self.cfg.
                                                  output_attentions)
            torch_attention_result, torch_qps, torch_time_consume = \
                test_helper.run_model(torch_model, use_cuda, num_iter, use_profile=False)
            print(
                f"BertAttention \"({batch_size},{seq_length:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            turbo_model = lambda: turbo_attention(input_tensor,
                                                  attention_mask,
                                                  output_attentions=self.cfg.
                                                  output_attentions)

            turbo_attention_result, turbo_qps, turbo_time_consume = \
                test_helper.run_model(turbo_model, use_cuda,
                                      num_iter)
            print(
                f"BertAttention \"({batch_size},{seq_length:03})\" ",
                f" {device} Turbo QPS, {turbo_qps}, time, {turbo_time_consume}"
            )

            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              turbo_attention_result[0])) < (
                                  1e-2 if use_cuda else 1e-4))
            # TODO(jiaruifang) result[1] won't be converted into torch tensor.
            # self.assertTrue(
            #     torch.max(
            #         torch.abs(torch_attention_result[1] -
            #                   turbo_attention_result[1])) < (
            #                       1e-2 if use_cuda else 1e-4))

            turbo_multiheaded_model = lambda: turbo_decoder_attention(
                input_tensor,
                input_tensor,
                input_tensor,
                attention_mask,
                layer_cache=None,
                attn_type="self",
                pre_layernorm=False,
                post_layernorm=True,
                post_add_input=False,
                is_trans_weight=False)
            turbo_decoder_attn_result, turbo_decoder_qps, turbo_decoder_time_consume = \
                test_helper.run_model(turbo_multiheaded_model, use_cuda,
                                      num_iter, use_profile=False)
            print(
                f"MultiHeadedAttention \"({batch_size},{seq_length:03})\" ",
                f" {device} Turbo QPS, {turbo_decoder_qps}, time, {turbo_decoder_time_consume}"
            )
            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              turbo_decoder_attn_result[0])) < (
                                  1e-3 if use_cuda else 1e-4))

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_bert_attention(self):
            self.check_torch_and_turbo(use_cuda=False, num_iter=1)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True, num_iter=1)

    globals()[f"TestBertAtt{batch_size}_{seq_length:3}"] = TestBertAttention


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 20, 40, 60, 80, 100]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
