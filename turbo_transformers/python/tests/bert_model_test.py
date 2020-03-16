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
import os
import contexttimer
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import turbo_transformers


class TestBertModel(unittest.TestCase):
    def setUp(self) -> None:
        model_id = os.path.join(os.path.dirname(__file__), 'test-model')
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        if not torch.cuda.is_available(
        ) or not turbo_transformers.config.is_with_cuda():
            self.test_device = torch.device('cpu:0')
            self.device = "CPU"
        else:
            self.test_device = torch.device('cuda:0')
            self.device = "GPU"

        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.torch_model = BertModel.from_pretrained(model_id)
        self.torch_model.eval()

        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

        self.ft_model = turbo_transformers.BertModel.from_pretrained(
            model_id, self.test_device)

    def test_bert_model(self):
        num_iter = 2
        input_ids = self.tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?')
        input_ids = torch.tensor([input_ids],
                                 dtype=torch.long,
                                 device=self.test_device)

        self.torch_model(input_ids)
        with contexttimer.Timer() as t:
            for it in range(num_iter):
                torch_result = self.torch_model(input_ids)
        print(f'BertModel Plain PyTorch QPS {num_iter / t.elapsed}')
        ft_result = self.ft_model(input_ids)
        with contexttimer.Timer() as t:
            for it in range(num_iter):
                ft_result = self.ft_model(input_ids)

        print(f'BertModel FastTransform QPS {num_iter / t.elapsed}')
        torch_result = (torch_result[0][:, 0]).cpu().numpy()
        ft_result = ft_result.cpu().numpy()

        self.assertTrue(
            numpy.allclose(torch_result, ft_result, atol=5e-3, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
