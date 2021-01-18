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
import enum
import time
import sys
import numpy as np


def test(use_cuda: bool):
    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')
    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_id)
    model.eval()
    model.to(test_device)
    torch.set_grad_enabled(False)

    cfg = model.config

    input_ids = torch.tensor(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=torch.long,
        device=test_device)
    # position_ids = torch.tensor(([1, 0, 0, 0], [1, 1, 1, 0]), dtype=torch.long, device = test_device)
    segment_ids = torch.tensor(([1, 1, 1, 0], [1, 0, 0, 0]),
                               dtype=torch.long,
                               device=test_device)

    start_time = time.time()
    for _ in range(10):
        torch_res = model(
            input_ids, token_type_ids=segment_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
    end_time = time.time()
    print("\ntorch time consum: {}".format(end_time - start_time))
    print("torch bert sequence output: ",
          torch_res[0][:, 0, :])  #get the first sequence
    print("torch bert pooler output: ", torch_res[1])  # pooled_output


if __name__ == "__main__":
    test(True)
    test(False)
    # test(LoadType.PRETRAINED, False)
