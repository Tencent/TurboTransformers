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
from transformers.modeling_bert import BertConfig, BertOutput


def create_shape_test(batch_size: int, seq_length: int):
    class TestBertOut(unittest.TestCase):
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
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)
            self.intermediate_size = self.cfg.intermediate_size  # 3072;
            self.hidden_size = self.cfg.hidden_size  # 768
            self.torch_bertout = BertOutput(self.cfg)
            self.torch_bertout.eval()
            if torch.cuda.is_available():
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

        def test_bertout(self):
            with open(f"gpu_bert_output_qps_{batch_size}_{seq_length:03}.txt",
                      "w") as of:
                num_steps = 2
                torch_result = self.torch_bertout(self.intermediate_output,
                                                  self.attention_output)
                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        torch_result = self.torch_bertout(
                            self.intermediate_output, self.attention_output)

                print(
                    f"BertOut({batch_size}, {seq_length:03}) Torch QPS {num_steps / t.elapsed}",
                    file=of)

                ft_result = self.ft_bertout(self.intermediate_output,
                                            self.attention_output)

                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        ft_result = self.ft_bertout(self.intermediate_output,
                                                    self.attention_output)

                print(
                    f"BertOut({batch_size}, {seq_length:03}) FastTransform QPS {num_steps / t.elapsed}",
                    file=of)
                self.assertTrue(
                    torch.max(torch.abs(torch_result - ft_result)) < 1e-4)

    TestBertOut.__name__ = f"TestBertOut_BatchSize_{batch_size}_SeqLen_{seq_length}"

    globals()[TestBertOut.__name__] = TestBertOut

    return TestBertOut


TestCases = [
    create_shape_test(batch_size=batch_size, seq_length=seq_length)
    for seq_length in (20, 40, 60, 80, 100, 120) for batch_size in (1, 2)
]

# TestBertOut = create_shape_test(batch_size=1, seq_length=20)

if __name__ == '__main__':
    # print(TestBertOut)
    unittest.main()
