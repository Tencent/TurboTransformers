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


class LoadType(enum.Enum):
    PYTORCH = "PYTORCH"
    PRETRAINED = "PRETRAINED"
    NPZ = "NPZ"


def test(loadtype: LoadType, use_cuda: bool):
    cfg = transformers.BertConfig()
    cfg.num_hidden_layers = 12
    model = transformers.BertModel(cfg)
    # model = transformers.BertModel.from_pretrained(model_id)
    model.eval()
    torch.set_grad_enabled(False)

    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')
    model.to(test_device)

    # cfg = model.config
    # use 4 threads for computing
    # turbo_transformers.set_num_threads(4)
    print(cfg)
    input_ids_list = []
    segment_ids_list = []

    for seq_len in [10, 20, 30, 40]:
        print(seq_len)
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(1, seq_len),
                                  dtype=torch.long,
                                  device=test_device)
        segment_ids_list.append(
            torch.zeros(size=(1, seq_len),
                        dtype=torch.long,
                        device=test_device))
        input_ids_list.append(input_ids)

    start_time = time.time()
    for input_ids, segment_ids in zip(input_ids_list, segment_ids_list):
        model(
            input_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
    end_time = time.time()
    print("\ntorch time consum: {}".format(end_time - start_time))

    # there are three ways to load pretrained model.
    if loadtype is LoadType.PYTORCH:
        # 1, from a PyTorch model, which has loaded a pretrained model
        # note that you can choose "turbo" or "onnxrt" as backend
        # "turbo" is a hand-crafted implementation and optimized with OMP.
        tt_model = turbo_transformers.BertModel.from_torch(model, test_device)
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
        tt_model = turbo_transformers.BertModel.from_npz(
            in_file, cfg, test_device)
    else:
        raise ("LoadType is not supported")

    start_time = time.time()
    for input_ids, segment_ids in zip(input_ids_list, segment_ids_list):
        tt_model(input_ids,
                 token_type_ids=segment_ids)  # sequence_output, pooled_output
    end_time = time.time()

    print("\nturbo time consum: {}".format(end_time - start_time))


if __name__ == "__main__":
    test(LoadType.PYTORCH, True)
    # test(LoadType.PRETRAINED, False)
