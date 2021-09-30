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
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaConfig
import numpy
import turbo_transformers
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper


class TestRobertaModel(unittest.TestCase):
    def init_data(self, use_cuda) -> None:
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        turbo_transformers.set_num_threads(4)

        self.test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')

        self.cfg = RobertaConfig()
        self.torch_model = RobertaModel(self.cfg)
        self.torch_model.eval()

        if torch.cuda.is_available():
            self.torch_model.to(self.test_device)

        self.turbo_model = turbo_transformers.RobertaModel.from_torch(
            self.torch_model, self.test_device)

    def check_torch_and_turbo(self, use_cuda):
        self.init_data(use_cuda)
        num_iter = 20
        device_name = "GPU" if use_cuda else "CPU"
        input_ids = torch.randint(low=0,
                                  high=self.cfg.vocab_size - 1,
                                  size=(1, 10),
                                  dtype=torch.long,
                                  device=self.test_device)

        torch_model = lambda: self.torch_model(input_ids)
        torch_result, torch_qps, torch_time = \
            test_helper.run_model(torch_model, use_cuda, num_iter)
        print(f'RobertaModel PyTorch({device_name}) QPS {torch_qps}')

        turbo_model = (lambda: self.turbo_model(input_ids))
        with turbo_transformers.pref_guard("roberta_perf") as perf:
            turbo_result, turbo_qps, turbo_time = \
                test_helper.run_model(turbo_model, use_cuda, num_iter)
        print(f'RobertaModel TurboTransformer({device_name}) QPS {turbo_qps}')

        torch_result_final = torch_result[0].cpu().numpy()

        turbo_result_final = turbo_result[0].cpu().numpy()
        # print(numpy.size(torch_result_final), numpy.size(turbo_result_final))
        # print(torch_result_final - turbo_result_final)
        self.assertTrue(
            numpy.allclose(torch_result_final,
                           turbo_result_final,
                           atol=1e-3,
                           rtol=1e-3))

    def test_Roberta_model(self):
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_torch_and_turbo(use_cuda=True)
        self.check_torch_and_turbo(use_cuda=False)


if __name__ == '__main__':
    unittest.main()
