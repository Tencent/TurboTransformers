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

import torch
import transformers
import turbo_transformers
import os

# use 4 threads for infernec
turbo_transformers.set_num_threads(4)
# load model from torch model
# model_id = os.path.join(os.path.dirname(__file__),
#                         '../../turbo_transformers/python/tests/test-model')
# model = transformers.BertModel.from_pretrained(model_id)

# load model from npz (recommend)
cfg = transformers.BertConfig()

model = turbo_transformers.BertModel.from_npz('../cpp/models/bert.npz', cfg)

model.eval()
cfg = model.config
batch_size = 2
seq_len = 128
torch.manual_seed(1)
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(batch_size, seq_len),
                          dtype=torch.long)
input_ids = torch.tensor(
    ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]), dtype=torch.long)
position_ids = torch.tensor(([1, 0, 0, 0], [1, 1, 1, 0]), dtype=torch.long)
segment_ids = torch.tensor(([1, 1, 1, 0], [1, 0, 0, 0]), dtype=torch.long)
torch.set_grad_enabled(False)
torch_res = model(
    input_ids, position_ids=position_ids, token_type_ids=segment_ids
)  # sequence_output, pooled_output, (hidden_states), (attentions)
print(torch_res[0][:, 0, :])
print(torch_res[1])
# tensor([[-1.4238,  1.0980, -0.3257,  ...,  0.7149, -0.3883, -0.1134],
#        [-0.8828,  0.6565, -0.6298,  ...,  0.2776, -0.4459, -0.2346]])

# there are two methods to load pretrained model.
# 1, from a torch model, which has loaded a pretrained model
tt_model = turbo_transformers.BertModelWithPooler.from_torch(model)
# 2. directly load from checkpoint (torch saved model)
# model = turbo_transformers.BertModel.from_pretrained(model_id)
res = tt_model(input_ids,
               position_ids=position_ids,
               token_type_ids=segment_ids)
print(res[0])
