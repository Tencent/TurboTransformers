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

from onmt.modules.position_ffn import PositionwiseFeedForward

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "ppf.txt"


def create_test(batch_size, input_len):
    class TestPositionwiseFeedForward(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            self.model_dim = 1024
            self.d_ff = 4096

            torch.set_grad_enabled(False)
            onmt_ffn = PositionwiseFeedForward(self.model_dim, self.d_ff)
            onmt_ffn.eval()
            if use_cuda:
                onmt_ffn.to(self.test_device)

            # (batch_size, input_len, model_dim)
            inputs = torch.rand(size=(batch_size, input_len, self.model_dim),
                                dtype=torch.float32,
                                device=self.test_device)
            return onmt_ffn, inputs

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            onmt_ffn, inputs = self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"
            # w_1.weight
            # w_1.bias
            # w_2.weight
            # w_2.bias
            # layer_norm.weight
            # layer_norm.bias

            for k, v in onmt_ffn.named_parameters():
                print(k)
            onmt_model = lambda: onmt_ffn(inputs)
            onmt_model_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter) # return output, attns

            print(
                f"ONMT PositionwiseFeedForward \"({batch_size}, {input_len:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            # self.assertTrue(
            #     torch.max(
            #         torch.abs(onmt_multi_headed_attention_result[0] -
            #                   turbo_self_attention_result)) < (
            #                       1e-3 if use_cuda else 1e-4))
            # with open(fname, "a") as fh:
            #     fh.write(
            #         f"\"({self.attn_type},{batch_size},{input_len:03})\", {torch_qps}, {turbo_qps}\n"
            #     )

        def test_multi_headed_attention(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestBertAtt{batch_size}_{input_len:3}"] = TestPositionwiseFeedForward


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")

for batch_size in [1, 2]:
    for input_len in [10, 16, 20, 30]:
        create_test(batch_size, input_len)

if __name__ == '__main__':
    unittest.main()
