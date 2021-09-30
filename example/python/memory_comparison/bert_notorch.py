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
import numpy as np
import transformers
import turbo_transformers
import enum
import time
import sys
import torch


def test(use_cuda: bool):
    test_device_name = "GPU" if use_cuda else "CPU"

    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')

    cfg = transformers.BertConfig()
    # use 4 threads for computing
    turbo_transformers.set_num_threads(4)

    input_ids = np.array(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=np.int64)
    segment_ids = np.array(([1, 1, 1, 0], [1, 0, 0, 0]), dtype=np.int64)

    input_ids_tensor = turbo_transformers.nparray2tensor(
        input_ids, test_device_name)
    segment_ids_tensor = turbo_transformers.nparray2tensor(
        segment_ids, test_device_name)
    # 3. load model from npz
    if len(sys.argv) == 2:
        try:
            print(sys.argv[1])
            in_file = sys.argv[1]
        except:
            sys.exit("ERROR. can not open ", sys.argv[1])
    else:
        in_file = "/home/jiaruifang/codes/TurboTransformers/bert.npz"
    # 255 MiB

    tt_model = turbo_transformers.BertModel.from_npz(in_file, cfg, test_device)

    # 1169 MiB
    start_time = time.time()
    for _ in range(10):
        res = tt_model(input_ids_tensor,
                       token_type_ids=segment_ids_tensor,
                       return_type=turbo_transformers.ReturnType.NUMPY
                       )  # sequence_output, pooled_output
    end_time = time.time()

    print("turbo bert sequence output:", res[0][:, 0, :])
    print("turbo bert pooler output: ", res[1])  # pooled_output
    print("\nturbo time consum: {}".format(end_time - start_time))


if __name__ == "__main__":
    test(True)
    test(False)
    # test(LoadType.PRETRAINED, False)
