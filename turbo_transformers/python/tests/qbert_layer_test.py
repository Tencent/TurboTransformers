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
from turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import time

model = transformers.BertModel.from_pretrained('bert-base-uncased')
model.eval()
torch.set_grad_enabled(False)

bertlayer = model.encoder.layer[0]
qbertlayer = turbo_transformers.QBertLayer.from_torch(bertlayer)
torchqbertlayer = torch.quantization.quantize_dynamic(bertlayer)

lens = [40, 60]
loops = 1

for l in lens:
    input_tensor = torch.rand((1, l, 768))
    attention_mask = torch.ones((1, l))
    attention_mask = attention_mask[:, None, None, :]
    attention_mask = (1.0 - attention_mask) * -10000.0
    print("seq length =", l)

    start = time.time()
    for i in range(loops):
        res = bertlayer(input_tensor, attention_mask, output_attentions=True)
    end = time.time()
    print("torch fp32 layer QPS =", loops / (end - start))

    start = time.time()
    for i in range(loops):
        res2 = qbertlayer(input_tensor, attention_mask, output_attentions=True)
    end = time.time()
    print("turbo fp32+int8 layer QPS =", loops / (end - start))

    start = time.time()
    for i in range(loops):
        res3 = torchqbertlayer(input_tensor,
                               attention_mask,
                               output_attentions=True)
    end = time.time()
    print("torch int8 layer QPS =", loops / (end - start))

print("max error against torch fp32 =", torch.max(torch.abs(res[0] - res2[0])))
print("max error against torch int8 =",
      torch.max(torch.abs(res3[0] - res2[0])))
print("max error between torch int8 and torch fp32 =",
      torch.max(torch.abs(res3[0] - res[0])))
