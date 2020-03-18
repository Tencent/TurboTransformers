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
import os
import contexttimer
import torch
import torch.jit
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertEncoder


class TestBertEncoder(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available(
        ) or not turbo_transformers.config.is_with_cuda():
            torch.set_num_threads(1)
            self.test_device = torch.device('cpu')
            self.device = "CPU"
        else:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.test_device = torch.device('cuda:0')
            self.device = "GPU"

        torch.set_grad_enabled(False)
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.path.dirname(__file__), 'test-model'))
        self.cfg = BertConfig(
            vocab_size_or_config_json_file=self.tokenizer.vocab_size)

        self.torch_encoder_layer = BertEncoder(self.cfg)
        self.torch_encoder_layer.eval()

        if torch.cuda.is_available():
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

    def test_bert_encoder(self):
        self.num_iter = 2

        ft_bert_layer_result = self.ft_bert_encoder(self.input_tensor,
                                                    self.attention_mask)
        if torch.cuda.is_available(
        ) and turbo_transformers.config.is_with_cuda():
            self.start.record()
        with contexttimer.Timer() as t:
            ft_bert_layer_result = None
            for it in range(self.num_iter):
                ft_bert_layer_result = self.ft_bert_encoder(
                    self.input_tensor,
                    self.attention_mask,
                    output=ft_bert_layer_result,
                    return_type=turbo_transformers.ReturnType.
                    turbo_transformers)
        if torch.cuda.is_available(
        ) and turbo_transformers.config.is_with_cuda():
            self.end.record()
            torch.cuda.synchronize()
            gpu_elapsed = self.start.elapsed_time(self.end) / 1e3
            print(
                f"BertEncoder FastTransform QPS, {self.num_iter / gpu_elapsed}, ",
                f"Time Cost, {gpu_elapsed / self.num_iter}")
        else:
            print(
                f"BertEncoder FastTransform QPS, {self.num_iter / t.elapsed}, ",
                f"Time Cost, {t.elapsed / self.num_iter}")

        ft_bert_layer_result = self.ft_bert_encoder(self.input_tensor,
                                                    self.attention_mask)

        torch_bert_layer_result = self.torch_encoder_layer(
            self.input_tensor, self.attention_mask,
            [None] * self.cfg.num_hidden_layers)

        if torch.cuda.is_available(
        ) and turbo_transformers.config.is_with_cuda():
            self.start.record()
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                torch_bert_layer_result = self.torch_encoder_layer(
                    self.input_tensor, self.attention_mask,
                    [None] * self.cfg.num_hidden_layers)
        if torch.cuda.is_available(
        ) and turbo_transformers.config.is_with_cuda():
            self.end.record()
            torch.cuda.synchronize()
            gpu_elapsed = self.start.elapsed_time(self.end) / 1e3

            print(f"BertEncoder Torch QPS, {self.num_iter / gpu_elapsed}, ",
                  f"Time Cost, {gpu_elapsed / self.num_iter}")
        else:
            print(f"BertEncoder Torch QPS, {self.num_iter / t.elapsed}, ",
                  f"Time Cost, {t.elapsed / self.num_iter}")
        diff = torch.abs(torch_bert_layer_result[0] - ft_bert_layer_result)
        self.assertTrue(torch.max(diff) < 1e-3)


if __name__ == '__main__':
    unittest.main()
