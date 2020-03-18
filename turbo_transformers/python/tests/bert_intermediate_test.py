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

import unittest

import sys
import torch
import turbo_transformers
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertIntermediate
import numpy
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


def create_test(batch_size, seq_length):
    class TestBertIntermediate(unittest.TestCase):
        def init_data(self, use_cuda: bool) -> None:
            if use_cuda:
                self.test_device = torch.device('cuda:0')
                self.device = "GPU"
            else:
                torch.set_num_threads(1)
                self.test_device = torch.device('cpu')
                self.device = "CPU"

            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)

            self.torch_intermediate = BertIntermediate(self.cfg)
            if torch.cuda.is_available():
                self.torch_intermediate.to(self.test_device)
            self.torch_intermediate.eval()

            self.ft_intermediate = turbo_transformers.BertIntermediate.from_torch(
                self.torch_intermediate)

        def check_torch_and_turbo(self, use_cuda):
            self.init_data(use_cuda=use_cuda)
            num_iter = 2
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32,
                                      device=self.test_device)

            ft_model = lambda: self.ft_intermediate(input_tensor)
            ft_result, ft_qps, ft_time = \
                test_helper.run_model(ft_model, use_cuda, num_iter)

            print(
                f"BertIntermediate \"({batch_size},{seq_length:03})\" ",
                f"{self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}")

            torch_model = lambda: self.torch_intermediate(input_tensor)
            torch_result, torch_qps, torch_time = \
                test_helper.run_model(torch_model, use_cuda, num_iter)

            print(
                f"BertIntermediate \"({batch_size},{seq_length:03})\" ",
                f"{self.device} Torch QPS,  {torch_qps}, time, {torch_time}")

            torch_result = torch_result.cpu().numpy()
            ft_result = ft_result.cpu().numpy()

            self.assertTrue(
                numpy.allclose(torch_result, ft_result, rtol=1e-4, atol=1e-3))

            with open("bert_intermediate_res.txt", "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {ft_qps}\n"
                )

        def test_intermediate(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestBertIntermediate_{batch_size}_{seq_length:03}"] = \
        TestBertIntermediate


with open("bert_intermediate_res.txt", "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
