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

import unittest
import torch
from transformers.modeling_albert import AlbertModel, AlbertConfig
import numpy
import turbo_transformers
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestAlbertModel(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        self.test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')

        self.cfg = AlbertConfig()
        self.torch_model = AlbertModel(self.cfg)
        self.torch_model.eval()

        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

        self.turbo_model = turbo_transformers.AlbertModel.from_torch(
            self.torch_model, self.cfg, self.test_device)

        self.turbo_pooler_model = turbo_transformers.AlbertModelWithPooler.from_torch(
            self.torch_model, self.cfg, self.test_device)

    def check_torch_and_turbo(self, use_cuda, use_pooler):
        self.init_data(use_cuda)
        num_iter = 2
        device_name = "GPU" if use_cuda else "CPU"
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(2, 32),
                                  dtype=torch.long,
                                  device=self.test_device)

        torch_model = lambda: self.torch_model(input_ids)
        torch_result, torch_qps, torch_time = \
            test_helper.run_model(torch_model, use_cuda, num_iter)
        print(f'AlbertModel Plain PyTorch({device_name}) QPS {torch_qps}')

        turbo_model = (
            lambda: self.turbo_pooler_model(input_ids)) if use_pooler else (
                lambda: self.turbo_model(input_ids))
        turbo_result, turbo_qps, turbo_time = \
            test_helper.run_model(turbo_model, use_cuda, num_iter)
        print(f'AlbertModel TurboTransformer({device_name}) QPS {turbo_qps}')

        torch_result_final = (torch_result[1]).cpu().numpy(
        ) if use_pooler else torch_result[0][:, 0].cpu().numpy()

        turbo_result_final = turbo_result[0].cpu().numpy()

        #TODO(jiaruifang, v_cshi) check why pooler introduce more difference
        if use_pooler:
            print(
                "encode output diff: ",
                numpy.max((torch_result[0][:, 0]).cpu().numpy() -
                          turbo_result[1].cpu().numpy()).reshape(-1))
            print(
                "pooler output diff: ",
                numpy.max(
                    (turbo_result_final - torch_result_final).reshape(-1)))
        (atol, rtol) = (1e-2, 1e-2) if use_pooler else (5e-3, 1e-4)

        self.assertTrue(
            numpy.allclose(torch_result_final,
                           turbo_result_final,
                           atol=atol,
                           rtol=rtol))

    def test_bert_model(self):
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True, use_pooler=False)
            self.check_torch_and_turbo(use_cuda=True, use_pooler=True)
        self.check_torch_and_turbo(use_cuda=False, use_pooler=False)
        self.check_torch_and_turbo(use_cuda=False, use_pooler=True)


if __name__ == '__main__':
    unittest.main()
