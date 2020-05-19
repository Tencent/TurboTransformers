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

fname = "ffn.txt"


def create_test(batch_size, input_len):
    class TestPositionwiseFeedForward(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            self.model_dim = 1024
            self.d_ff = 4096

            torch.set_grad_enabled(False)
            onmt_ffn = PositionwiseFeedForward(self.model_dim, self.d_ff)
            onmt_ffn.eval()
            if use_cuda:
                onmt_ffn.to(self.test_device)

            turbo_ffn_trans = turbo_transformers.PositionwiseFeedForward.from_onmt(
                onmt_ffn, is_trans_weight=True)
            turbo_ffn_notrans = turbo_transformers.PositionwiseFeedForward.from_onmt(
                onmt_ffn, is_trans_weight=False)
            # (batch_size, input_len, model_dim)
            inputs = torch.rand(size=(batch_size, input_len, self.model_dim),
                                dtype=torch.float32,
                                device=self.test_device)
            return onmt_ffn, turbo_ffn_trans, turbo_ffn_notrans, inputs

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            onmt_ffn, turbo_ffn_trans, turbo_ffn_notrans, inputs = self.init_data(
                use_cuda)
            device = "GPU" if use_cuda else "CPU"
            onmt_model = lambda: onmt_ffn(inputs)
            onmt_model_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter)

            print(
                f"PositionwiseFeedForward \"({batch_size}, {input_len:03})\" ",
                f"{device} ONMT QPS, {torch_qps}, time, {torch_time_consume}")

            turbo_model_trans = lambda: turbo_ffn_trans(inputs,
                                                        is_trans_weight=True)
            with turbo_transformers.pref_guard("gpref_test") as perf:
                turbo_model_result, turbo_qps_trans, turbo_time_consume_trans = \
                    test_helper.run_model(turbo_model_trans, use_cuda, num_iter)

            print(
                f"PositionwiseFeedForward \"({batch_size}, {input_len:03})\" ",
                f"{device} Turbo Trans QPS, {turbo_qps_trans}, time, {turbo_time_consume_trans}"
            )

            turbo_model_notrans = lambda: turbo_ffn_notrans(
                inputs, is_trans_weight=False)
            with turbo_transformers.pref_guard("gpref_test") as perf:
                turbo_model_result, turbo_qps_notrans, turbo_time_consume_notrans = \
                    test_helper.run_model(turbo_model_notrans, use_cuda, num_iter)

            print(
                f"PositionwiseFeedForward Notrans \"({batch_size}, {input_len:03})\" ",
                f"{device} Turbo NoTrans QPS, {turbo_qps_notrans}, time, {turbo_time_consume_notrans}"
            )
            self.assertTrue(
                torch.max(torch.abs(turbo_model_result - onmt_model_result)) <
                (1e-3 if use_cuda else 1e-4))

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{input_len:03})\", {torch_qps}, {turbo_qps_trans}, {turbo_qps_notrans}\n"
                )

        def test_positionwise_feed_forward(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestPositionwiseFeedForward{batch_size}_{input_len:3}"] = TestPositionwiseFeedForward


with open(fname, "w") as fh:
    fh.write(", torch, turbo_trans, turbo_notrans\n")

for batch_size in [4]:
    for input_len in [10, 20, 30, 40, 50]:
        create_test(batch_size, input_len)

if __name__ == '__main__':
    unittest.main()
