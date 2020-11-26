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
import sys


class LoadType(enum.Enum):
    PYTORCH = "PYTORCH"
    PRETRAINED = "PRETRAINED"
    NPZ = "NPZ"


def test(loadtype: LoadType, use_cuda: bool):
    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')
    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_id)
    model.eval()
    model.to(test_device)
    torch.set_grad_enabled(False)

    cfg = model.config
    # use 4 threads for computing
    turbo_transformers.set_num_threads(4)

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

    # there are three ways to load pretrained model.
    if loadtype is LoadType.PYTORCH:
        # 1, from a PyTorch model, which has loaded a pretrained model
        # note that you can choose "turbo" or "onnxrt" as backend
        # "turbo" is a hand-crafted implementation and optimized with OMP.
        tt_model = turbo_transformers.BertModel.from_torch(
            model, test_device, "turbo")
    elif loadtype is LoadType.PRETRAINED:
        # 2. directly load from checkpoint (torch saved model)
        tt_model = turbo_transformers.BertModel.from_pretrained(
            model_id, test_device)
    elif loadtype is LoadType.NPZ:
        # 3. load model from npz
        if len(sys.argv) == 2:
            try:
                print(sys.argv[1])
                in_file = sys.argv[1]
            except:
                sys.exit("ERROR. can not open ", sys.argv[1])
        else:
            in_file = "/workspace/bert_torch.npz"
        tt_model = turbo_transformers.BertModel.from_npz(in_file, cfg)
    else:
        raise ("LoadType is not supported")

    start_time = time.time()
    for _ in range(10):
        res = tt_model(
            input_ids,
            token_type_ids=segment_ids)  # sequence_output, pooled_output
    end_time = time.time()

    print("turbo bert sequence output:", res[0][:, 0, :])
    print("turbo bert pooler output: ", res[1])  # pooled_output
    print("\nturbo time consum: {}".format(end_time - start_time))
    assert (torch.max(torch.abs(res[0] - torch_res[0])) < 0.2)


if __name__ == "__main__":
    test(LoadType.NPZ, False)
    # test(LoadType.PRETRAINED, False)
