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
import torch
import transformers
import turbo_transformers
import enum
import time
import numpy


class LoadType(enum.Enum):
    PYTORCH = "PYTORCH"
    PRETRAINED = "PRETRAINED"
    NPZ = "NPZ"


def test(loadtype: LoadType, use_cuda: bool):
    cfg = transformers.GPT2Config()
    model = transformers.GPT2Model(cfg)
    model.eval()
    torch.set_grad_enabled(False)

    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')

    cfg = model.config
    # use 4 threads for computing
    turbo_transformers.set_num_threads(4)

    input_ids = torch.tensor(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=torch.long)

    start_time = time.time()
    for _ in range(10):
        torch_res = model(input_ids)
    end_time = time.time()
    print("\ntorch time consum: {}".format(end_time - start_time))

    # there are three ways to load pretrained model.
    if loadtype is LoadType.PYTORCH:
        # 1, from a PyTorch model, which has loaded a pretrained model
        tt_model = turbo_transformers.GPT2Model.from_torch(model, test_device)
    else:
        raise ("LoadType is not supported")

    start_time = time.time()
    for _ in range(10):
        res = tt_model(input_ids)  # sequence_output, pooled_output
    end_time = time.time()

    print("\nturbo time consum: {}".format(end_time - start_time))
    assert (numpy.max(
        numpy.abs(res[0].cpu().numpy() - torch_res[0].cpu().numpy())) < 0.1)


if __name__ == "__main__":
    test(LoadType.PYTORCH, use_cuda=False)
