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
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel
import numpy
import turbo_transformers
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestBertModel(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        model_id = os.path.join(os.path.dirname(__file__), 'test-model')
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        self.test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')

        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.torch_model = BertModel.from_pretrained(model_id)
        self.torch_model.eval()

        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

        self.turbo_model = turbo_transformers.BertModel.from_pretrained(
            model_id, self.test_device)

        self.turbo_pooler_model = turbo_transformers.BertModelWithPooler.from_pretrained(
            model_id, self.test_device)

    def check_torch_and_turbo(self, use_cuda, use_pooler):
        self.init_data(use_cuda)
        num_iter = 2
        device = "GPU" if use_cuda else "CPU"
        input_ids = self.tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?')
        input_ids = torch.tensor([input_ids],
                                 dtype=torch.long,
                                 device=self.test_device)

        torch_model = lambda: self.torch_model(input_ids)
        torch_result, torch_qps, torch_time = \
            test_helper.run_model(torch_model, use_cuda, num_iter)
        print(f'BertModel Plain PyTorch({device}) QPS {torch_qps}')

        turbo_model = (
            lambda: self.turbo_model(input_ids)) if use_pooler else (
                lambda: self.turbo_pooler_model(input_ids))
        turbo_result, turbo_qps, turbo_time = \
            test_helper.run_model(turbo_model, use_cuda, num_iter)
        print(f'BertModel FastTransform({device}) QPS {turbo_qps}')

        torch_result = (
            torch_result[0][:, 0]).cpu().numpy() if use_pooler else (
                torch_result[1]).cpu().numpy()
        turbo_result = turbo_result.cpu().numpy()

        self.assertTrue(
            numpy.allclose(torch_result, turbo_result, atol=5e-3, rtol=1e-4))

    def test_bert_model(self):
        self.check_torch_and_turbo(use_cuda=False, use_pooler=True)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True, use_pooler=True)
        self.check_torch_and_turbo(use_cuda=False, use_pooler=False)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=False, use_pooler=False)


if __name__ == '__main__':
    unittest.main()
