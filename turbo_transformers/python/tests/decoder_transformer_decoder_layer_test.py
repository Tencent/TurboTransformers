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

from onmt.decoders.transformer import TransformerDecoderLayer

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size, src_length, T):
    class TestDecoder(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.model_dim = 1024
            self.onmt_decoder = TransformerDecoderLayer(d_model=self.model_dim,
                                                        heads=8,
                                                        d_ff=32,
                                                        dropout=0.,
                                                        attention_dropout=0.)
            self.onmt_decoder.eval()
            if use_cuda:
                self.onmt_decoder.to(test_device)

            self.turbo_decoder = turbo_transformers.TransformerDecoderLayer.from_onmt(
                self.onmt_decoder)

        def check_torch_and_turbo(self, use_cuda, num_iter=2):
            if use_cuda:
                return
            deivce_type = "GPU" if use_cuda else "CPU"
            info = f"\"({deivce_type}, {batch_size}, {src_length}, {T})\""

            self.init_data(use_cuda=use_cuda)

            self.inputs = torch.rand(batch_size,
                                     T,
                                     self.model_dim,
                                     dtype=torch.float32,
                                     device=self.test_device)
            self.memory_bank = torch.rand(batch_size,
                                          src_length,
                                          self.model_dim,
                                          dtype=torch.float32,
                                          device=self.test_device)

            self.src_pad_mask = torch.ones(batch_size,
                                           1,
                                           src_length,
                                           dtype=torch.float32,
                                           device=self.test_device)
            self.tgt_pad_mask = torch.ones(batch_size,
                                           1,
                                           T,
                                           dtype=torch.float32,
                                           device=self.test_device)

            onmt_model = lambda: self.onmt_decoder(self.inputs,
                                                   self.memory_bank,
                                                   self.src_pad_mask.bool(),
                                                   self.tgt_pad_mask.bool(),
                                                   layer_cache=None,
                                                   step=None,
                                                   future=False)

            onmt_result, torch_qps, torch_time_consume = \
                test_helper.run_model(onmt_model, use_cuda, num_iter)

            onmt_mid, attns, attn_align = onmt_result

            print(
                f"ONMT n {info} ",
                f"{deivce_type} Torch QPS, {torch_qps}, time, {torch_time_consume}"
            )

            turbo_model = lambda: self.turbo_decoder(self.inputs,
                                                     self.memory_bank,
                                                     self.src_pad_mask,
                                                     self.tgt_pad_mask,
                                                     layer_cache=None,
                                                     step=None,
                                                     future=False)
            turbo_result, torch_qps, torch_time_consume = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)

            turbo_mid, turbo_attns, _ = turbo_result

            print(
                f"Turbo n {info} ",
                f"{deivce_type} Torch QPS, {torch_qps}, time, {torch_time_consume}"
            )

            self.assertTrue(
                torch.max(torch.abs(onmt_mid -
                                    turbo_mid)) < (1e-3 if use_cuda else 1e-4))
            self.assertTrue(
                torch.max(torch.abs(attns - turbo_attns)) < (
                    1e-3 if use_cuda else 1e-4))

        def test_decoder(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestDecoder{batch_size}_{src_length}_{T}"] = TestDecoder


for batch_size in [1, 2]:
    for src_length in [10, 20]:
        for T in [1, 2]:
            create_test(batch_size, src_length, T)

if __name__ == '__main__':
    unittest.main()
