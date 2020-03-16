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
import torch.onnx
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertLayer

fname = "ft_bertlayer.txt"


def create_test(batch_size, seq_length):
    class TestBertLayer(unittest.TestCase):
        def setUp(self) -> None:
            if not torch.cuda.is_available(
            ) or not turbo_transformers.config.is_with_cuda():
                torch.set_num_threads(1)
                self.test_device = torch.device('cpu')
                self.device = "CPU"
            else:
                self.test_device = torch.device('cuda:0')
                self.device = "GPU"

            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0)

            self.torch_bert_layer = BertLayer(self.cfg)
            self.torch_bert_layer.eval()
            if torch.cuda.is_available():
                self.torch_bert_layer.to(self.test_device)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32,
                                           device=self.test_device)

            self.attention_mask = torch.ones((batch_size, seq_length),
                                             dtype=torch.float32,
                                             device=self.test_device)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

            self.ft_bert_layer = turbo_transformers.BertLayer.from_torch(
                self.torch_bert_layer)

        def test_bert_layer(self):
            num_iter = 2

            torch_bert_layer_result = self.torch_bert_layer(
                self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_bert_layer_result = self.torch_bert_layer(
                        self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                torch_elapsed = start.elapsed_time(end) / 1e3

            if torch.cuda.is_available():
                self.torch_qps = num_iter / torch_elapsed
                self.torch_time = torch_elapsed / num_iter
            else:
                self.torch_qps = num_iter / t.elapsed
                self.torch_time = t.elapsed / num_iter

            print(
                f"BertLayer \"({batch_size},{seq_length:03})\" {self.device} Torch QPS,  {self.torch_qps}, time, {self.torch_time}"
            )

            ft_bert_layer_result = self.ft_bert_layer(self.input_tensor,
                                                      self.attention_mask)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                ft_elapsed = 1e-3
                ft_result = None
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    ft_bert_layer_result = self.ft_bert_layer(
                        self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                # in ms, rescale to sec
                ft_elapsed = start.elapsed_time(end) / 1e3

            ft_qps = 0
            ft_time = 0
            if torch.cuda.is_available():
                ft_qps = num_iter / ft_elapsed
                ft_time = ft_elapsed / num_iter
            else:
                ft_qps = num_iter / t.elapsed
                ft_time = t.elapsed / num_iter

            print(
                f"BertLayer \"({batch_size},{seq_length:03})\" {self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}"
            )

            self.assertTrue(
                torch.max(
                    torch.abs(torch_bert_layer_result[0] -
                              ft_bert_layer_result)) < 1e-3)
            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {self.torch_qps}, {ft_qps}\n"
                )

    globals()[f"TestBertLayer{batch_size}_{seq_length:03}"] = TestBertLayer


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 20]:
    for seq_length in [10, 20, 40, 60, 80, 120, 200, 300, 400, 500]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
