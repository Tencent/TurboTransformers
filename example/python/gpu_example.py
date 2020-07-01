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
import numpy
import torch
import transformers
from transformers import BertTokenizer
import turbo_transformers

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    test_device = torch.device('cuda:0')
    model_id = "bert-base-uncased"
    model_torch = transformers.BertModel.from_pretrained(model_id)
    model_torch.eval()
    model_torch.to(test_device)
    cfg = model_torch.config
    # there are three methods to load pretrained model.
    # 1. load model from checkpoint in file
    # model_tt = turbo_transformers.BertModel.from_pretrained(model_id, test_device)
    # 2. load model from pytorch model
    model_tt = turbo_transformers.BertModel.from_torch(model_torch,
                                                       test_device)
    # 3. load from npz file
    # model_tt = turbo_transformers.BertModel.from_npz(model_torch, cfg, test_device)

    tokenizer = BertTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.encode('测试一下哈')
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=test_device)
    torch_result = model_torch(input_ids)
    torch_result = (torch_result[0][:, 0]).cpu().numpy()
    # print(torch_result)

    tt_result = model_tt(input_ids)
    tt_result = tt_result[0].cpu().numpy()
    # print(tt_result - torch_result)
    print("max diff between turbo and torch",
          numpy.max(numpy.abs(tt_result - torch_result)))
