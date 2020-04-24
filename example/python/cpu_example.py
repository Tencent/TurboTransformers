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
import os
import numpy
import sys

if __name__ == "__main__":
    cfg = transformers.BertConfig()
    # use 4 threads for computing
    turbo_transformers.set_num_threads(4)
    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_id)

    model.eval()
    cfg = model.config
    batch_size = 2
    seq_len = 54
    torch.manual_seed(1)
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long)
    input_ids = torch.tensor(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=torch.long)
    position_ids = torch.tensor(([1, 0, 0, 0], [1, 1, 1, 0]), dtype=torch.long)
    segment_ids = torch.tensor(([1, 1, 1, 0], [1, 0, 0, 0]), dtype=torch.long)
    torch.set_grad_enabled(False)
    torch_res = model(
        input_ids, position_ids=position_ids, token_type_ids=segment_ids
    )  # sequence_output, pooled_output, (hidden_states), (attentions)
    torch_seqence_output = torch_res[0][:, 0, :]
    print(torch_res[0][:, 0, :])
    print(torch_res[1])  #pooled_output

    # there are three ways to load pretrained model.
    # 1, from a PyTorch model, which has loaded a pretrained model
    # tt_model = turbo_transformers.BertModelWithPooler.from_torch(model)
    # 2. directly load from checkpoint (torch saved model)
    # tt_model = turbo_transformers.BertModelWithPooler.from_pretrained(model_id)
    # 3. load model from npz
    if len(sys.argv) == 2:
        try:
            print(sys.argv[1])
            in_file = sys.argv[1]
        except:
            sys.exit("ERROR. can not open ", sys.argv[1])
    else:
        in_file = "/workspace/bert_torch.npz"
    tt_model = turbo_transformers.BertModelWithPooler.from_npz(in_file, cfg)

    res = tt_model(input_ids,
                   position_ids=position_ids,
                   token_type_ids=segment_ids)  #pooled_output, sequence_output
    tt_seqence_output = res[1]
    print(res[1])
    print(res[0])  #pooled_output
    assert (torch.max(torch.abs(tt_seqence_output - torch_seqence_output)) <
            0.1)
