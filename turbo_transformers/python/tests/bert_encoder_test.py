# Copyright 2020 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import turbo_transformers

import unittest
import sys
import torch
import torch.jit
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertEncoder
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestBertEncoder(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        if use_cuda:
            self.test_device = torch.device('cuda:0')
        else:
            torch.set_num_threads(1)
            self.test_device = torch.device('cpu')

        torch.set_grad_enabled(False)
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.path.dirname(__file__), 'test-model'))
        self.cfg = BertConfig(
            vocab_size_or_config_json_file=self.tokenizer.vocab_size)

        self.torch_encoder_layer = BertEncoder(self.cfg)
        self.torch_encoder_layer.eval()

        if use_cuda:
            self.torch_encoder_layer.to(self.test_device)

        self.batch_size = 1
        self.seq_length = 40
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length,
                                             self.hidden_size),
                                       dtype=torch.float32,
                                       device=self.test_device)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length),
                                         dtype=torch.float32,
                                         device=self.test_device)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        self.ft_bert_encoder = turbo_transformers.BertEncoder.from_torch(
            self.torch_encoder_layer)

    def check_torch_and_turbo(self, use_cuda=True):
        self.init_data(use_cuda=use_cuda)
        self.num_iter = 2

        ft_bert_layer_result = None
        ft_model = lambda: self.ft_bert_encoder(self.input_tensor,
                                                self.attention_mask,
                                                output=ft_bert_layer_result,
                                                return_type=turbo_transformers.
                                                ReturnType.turbo_transformers)

        ft_bert_layer_result, turbo_qps, turbo_time_consume = \
            test_helper.run_model(ft_model, use_cuda, self.num_iter)

        print(f"BertEncoder FastTransform QPS, {turbo_qps}, ",
              f"Time Cost, {turbo_time_consume}")

        ft_bert_layer_result = self.ft_bert_encoder(self.input_tensor,
                                                    self.attention_mask)

        torch_model = lambda: self.torch_encoder_layer(
            self.input_tensor, self.attention_mask, [None] * self.cfg.
            num_hidden_layers)

        torch_bert_layer_result, torch_qps, torch_time_consume = \
            test_helper.run_model(torch_model, use_cuda, self.num_iter)

        print(f"BertEncoder Torch QPS, {torch_qps}, ",
              f"Time Cost, {torch_time_consume}")

        diff = torch.abs(torch_bert_layer_result[0] - ft_bert_layer_result)
        self.assertTrue(torch.max(diff) < 1e-3)

    def test_embedding(self):
        self.check_torch_and_turbo(use_cuda=False)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_with_cuda():
            self.check_torch_and_turbo(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
