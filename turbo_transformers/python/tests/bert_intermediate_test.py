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

import contexttimer
import time
import torch
import turbo_transformers
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertIntermediate
import numpy
import os


def create_test(batch_size, seq_length):
    class TestBertIntermediate(unittest.TestCase):
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

            self.torch_intermediate = BertIntermediate(self.cfg)
            if torch.cuda.is_available():
                self.torch_intermediate.to(self.test_device)
            self.torch_intermediate.eval()

            self.ft_intermediate = turbo_transformers.BertIntermediate.from_torch(
                self.torch_intermediate)

        def test_intermediate(self):
            num_iter = 2
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32,
                                      device=self.test_device)

            # warmup
            ft_result = self.ft_intermediate(input_tensor)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                ft_elapsed = 0.
                start.record()

            with contexttimer.Timer() as t:
                ft_result = None
                for it in range(num_iter):
                    ft_result = self.ft_intermediate(
                        input_tensor,
                        output=ft_result,
                        return_type=turbo_transformers.ReturnType.
                        turbo_transformers)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                # in ms, rescale to sec
                ft_elapsed = start.elapsed_time(end) / 1e3

            # get torch result
            ft_result = self.ft_intermediate(input_tensor)

            ft_qps = 0
            ft_time = 0
            if torch.cuda.is_available():
                ft_qps = num_iter / ft_elapsed
                ft_time = ft_elapsed / num_iter
            else:
                ft_qps = num_iter / t.elapsed
                ft_time = t.elapsed / num_iter

            print(
                f"BertIntermediate \"({batch_size},{seq_length:03})\" {self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}"
            )

            # warmup
            torch_result = self.torch_intermediate(input_tensor)
            torch_elapsed = 0.

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_result = self.torch_intermediate(input_tensor)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                torch_elapsed = start.elapsed_time(end) / 1e3
                torch_qps = num_iter / torch_elapsed
                torch_time = torch_elapsed / num_iter
            else:
                torch_qps = num_iter / t.elapsed
                torch_time = t.elapsed / num_iter

            print(
                f"BertIntermediate \"({batch_size},{seq_length:03})\" {self.device} Torch QPS,  {torch_qps}, time, {torch_time}"
            )
            torch_result = torch_result.cpu().numpy()
            ft_result = ft_result.cpu().numpy()

            self.assertTrue(
                numpy.allclose(torch_result, ft_result, rtol=1e-4, atol=1e-3))

            with open("bert_intermediate_res.txt", "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {ft_qps}\n"
                )

    globals(
    )[f"TestBertIntermediate_{batch_size}_{seq_length:03}"] = TestBertIntermediate


with open("bert_intermediate_res.txt", "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
