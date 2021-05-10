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
import sys
import turbo_transformers

import unittest
import torch
import os

from onmt.encoders.transformer import TransformerEncoderLayer

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_transformer_encoder_layer.txt"


def create_test(batch_size, src_length, T, with_quantize_dynamic=False):
    class TestEncoder(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.model_dim = 1024
            self.onmt_encoder = TransformerEncoderLayer(d_model=self.model_dim,
                                                        heads=8,
                                                        d_ff=1024,
                                                        dropout=0.,
                                                        attention_dropout=0.)
            self.onmt_encoder.eval()
            if use_cuda:
                self.onmt_encoder.to(self.test_device)
            self.turbo_encoder = turbo_transformers.TransformerEncoderLayer.from_onmt(
                self.onmt_encoder)

            # https://pytorch.org/docs/stable/quantization.html
            if with_quantize_dynamic and not use_cuda:
                self.quantized_onmt_encoder = torch.quantization.quantize_dynamic(
                    self.onmt_encoder)

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            deivce_type = "GPU" if use_cuda else "CPU"
            info = f"\"({deivce_type}, {batch_size}, {src_length}, {T})\""

            self.init_data(use_cuda=use_cuda)

            self.inputs = torch.rand(
                                    batch_size,
                                    src_length,
                                    self.model_dim,
                                    dtype=torch.float32,
                                    device=self.test_device)

            self.mask = torch.randint(-100, 0, (batch_size, 1, src_length), dtype=torch.int64, device=self.test_device)

            onmt_mask = self.mask > 0

            onmt_model = lambda: self.onmt_encoder(self.inputs, onmt_mask)

            onmt_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter)

            print(
                f"ONMT Encoder {info} ",
                f"{deivce_type} QPS, {torch_qps}, time, {torch_time_consume}")

            if with_quantize_dynamic and not use_cuda:
                quantized_onmt_model = lambda: self.quantized_onmt_encoder(
                    self.inputs,onmt_mask)

                quantized_onmt_result, quantized_torch_qps, quantized_torch_time_consume = \
                    test_helper.run_model(quantized_onmt_model, use_cuda, num_iter)

                print(
                    f"ONMT Quantized Encoder {info} ",
                    f"{deivce_type} QPS, {quantized_torch_qps}, time, {quantized_torch_time_consume}"
                )

            turbo_model = lambda: self.turbo_encoder(self.inputs, onmt_mask)

            with turbo_transformers.pref_guard(info) as perf:
                turbo_result, turbo_qps, turbo_time_consume = \
                    test_helper.run_model(turbo_model, use_cuda, num_iter)

            print(
                f"Turbo Encoder {info} ",
                f"{deivce_type} QPS, {turbo_qps}, time, {turbo_time_consume}")

            print(f"diff max {torch.max(torch.abs(onmt_result - turbo_result))}")
            self.assertTrue(
                torch.max(torch.abs(onmt_result - turbo_result)) < (1e-3 if use_cuda else 1e-4))

            if with_quantize_dynamic and not use_cuda:
                with open(fname, "a") as fh:
                    fh.write(
                        f"{info} {torch_qps}, {quantized_torch_qps}, {turbo_qps}\n"
                    )
            else:
                with open(fname, "a") as fh:
                    fh.write(f"{info} {torch_qps}, {turbo_qps}\n")

        def test_encoder(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestEncoder{batch_size}_{src_length}_{T}_{with_quantize_dynamic}"] = TestEncoder


with open(fname, "w") as fh:
    fh.write(", torch, *q_torch, turbo_transformers\n")

for quantize in [True]:
    for batch_size in [4]:
        for src_length in [10, 20, 40, 60, 80, 100]:
            for T in range(10, src_length, 10):
                create_test(batch_size, src_length, T, quantize)

if __name__ == '__main__':
    unittest.main()
