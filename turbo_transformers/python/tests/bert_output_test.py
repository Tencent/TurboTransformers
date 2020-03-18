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
import io
import torch
import torch.jit
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertOutput
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_shape_test(batch_size: int, seq_length: int):
    class TestBertOut(unittest.TestCase):
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
            self.intermediate_size = self.cfg.intermediate_size  # 3072;
            self.hidden_size = self.cfg.hidden_size  # 768
            self.torch_bertout = BertOutput(self.cfg)
            self.torch_bertout.eval()
            if use_cuda:
                self.torch_bertout.to(self.test_device)

            self.ft_bertout = turbo_transformers.BertOutput.from_torch(
                self.torch_bertout)

            self.intermediate_output = torch.rand(
                size=(batch_size, seq_length, self.intermediate_size),
                dtype=torch.float32,
                device=self.test_device)
            self.attention_output = torch.rand(size=(batch_size, seq_length,
                                                     self.hidden_size),
                                               dtype=torch.float32,
                                               device=self.test_device)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda)
            sio = io.StringIO()
            num_iter = 2
            device = "GPU" if use_cuda else "CPU"

            torch_model = lambda: self.torch_bertout(self.intermediate_output,
                                                     self.attention_output)
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)
            print(f'BertModel Plain PyTorch({device}) QPS {torch_qps}',
                  file=sio)

            turbo_model = lambda: self.ft_bertout(self.intermediate_output,
                                                  self.attention_output)
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)
            print(f'BertModel Plain FastTransform({device}) QPS {turbo_qps}',
                  file=sio)

            self.assertTrue(
                torch.max(torch.abs(torch_result - turbo_result)) < 1e-4)

            sio.seek(0)
            with open(f"gpu_bert_output_qps_{batch_size}_{seq_length:03}.txt",
                      "w") as of:
                for line in sio:
                    print(line.strip(), file=of)

        def test_bertout(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    TestBertOut.__name__ = f"TestBertOut_BatchSize_{batch_size}_SeqLen_{seq_length}"
    globals()[TestBertOut.__name__] = TestBertOut


for seq_length in (20, 40, 60, 80, 100, 120):
    for batch_size in (1, 2):
        create_shape_test(batch_size=batch_size, seq_length=seq_length)

if __name__ == '__main__':
    unittest.main()
