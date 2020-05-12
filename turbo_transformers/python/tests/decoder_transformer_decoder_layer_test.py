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
# from onmt.translate.translator import Translator

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size, src_length, T):
    class TestDecoder(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.onmt_decoder = TransformerDecoderLayer(d_model=1024,
                                                        heads=8,
                                                        d_ff=1,
                                                        dropout=0.,
                                                        attention_dropout=0.)
            self.onmt_decoder.eval()
            if use_cuda:
                self.onmt_decoder.to(test_device)

            self.turbo_decoder = turbo_transformers.TransformerDecoderLayer.from_onmt(
                self.onmt_decoder)

        def check_torch_and_turbo(self, use_cuda, num_iter=2):
            self.init_data(use_cuda=use_cuda)
            model_dim = 1024
            T = 1
            src_length = 20
            self.inputs = torch.rand(batch_size,
                                     T,
                                     model_dim,
                                     dtype=torch.float32,
                                     device=self.test_device)
            self.memory_bank = torch.rand(batch_size,
                                          src_length,
                                          model_dim,
                                          dtype=torch.float32,
                                          device=self.test_device)
            self.src_pad_mask = torch.zeros(batch_size,
                                            1,
                                            src_length,
                                            dtype=torch.float32,
                                            device=self.test_device)
            self.tgt_pad_mask = torch.zeros(batch_size,
                                            1,
                                            T,
                                            dtype=torch.float32,
                                            device=self.test_device)

            onmt_mid, attns, attn_align = self.onmt_decoder(
                self.inputs,
                self.memory_bank,
                self.src_pad_mask.bool(),
                self.tgt_pad_mask.bool(),
                layer_cache=None,
                step=None,
                future=False)

            turbo_mid, turbo_attns, _ = self.turbo_decoder(self.inputs,
                                                           self.memory_bank,
                                                           self.src_pad_mask,
                                                           self.tgt_pad_mask,
                                                           layer_cache=None,
                                                           step=None,
                                                           future=False)

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
