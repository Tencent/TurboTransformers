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


def test(use_cuda):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    turbo_transformers.set_num_threads(4)

    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')

    cfg = RobertaConfig()
    torch_model = RobertaModel(cfg)
    torch_model.eval()

    if torch.cuda.is_available():
        torch_model.to(test_device)

    turbo_model = turbo_transformers.RobertaModel.from_torch(
        torch_model, test_device)

    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(1, 10),
                              dtype=torch.long,
                              device=test_device)

    torch_result = torch_model(input_ids)
    torch_result_final = torch_result[0].cpu().numpy()

    turbo_result = turbo_model(input_ids)
    turbo_result_final = turbo_result[0].cpu().numpy()

    # See the differences
    # print(numpy.size(torch_result_final), numpy.size(turbo_result_final))
    print(torch_result_final - turbo_result_final)
    assert (numpy.allclose(torch_result_final,
                           turbo_result_final,
                           atol=1e-3,
                           rtol=1e-3))


if __name__ == "__main__":
    test(use_cuda=False)
