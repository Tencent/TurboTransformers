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

from dynamic_allocator import trunked_greedy_by_size_offset_calculation
from static_allocator import greedy_by_size_offset_calculation
import random

if __name__ == "__main__":
    from bert_tensor_usage import get_bert_tensor_usage_record
    random.seed(0)

    length_list = []
    ours_cost_list = []
    baseline_cost_list = []
    ours_acc_new_allocate_list = []
    baseline_acc_new_allocate_list = []
    acc_ours_new_allocate = 0.0
    acc_baseline_new_allocate = 0.0
    with open("random_len.txt", "r") as infile:
        for line in infile.readlines():
            length = int(line)
            print(f"begin schedule for allocate {length}")
            tur = get_bert_tensor_usage_record(1, length, 1)
            _, _, _, res1 = trunked_greedy_by_size_offset_calculation(
                tur, True)
            _, res2 = greedy_by_size_offset_calculation(tur, False)
            ours_cost, ours_new_allocate = res1[0], res1[1]
            basline_cost = res2
            length_list.append(length)
            ours_cost_list.append(ours_cost)
            baseline_cost_list.append(basline_cost)
            acc_ours_new_allocate += ours_new_allocate
            acc_baseline_new_allocate += basline_cost
            ours_acc_new_allocate_list.append(acc_ours_new_allocate)
            baseline_acc_new_allocate_list.append(acc_baseline_new_allocate)

    with open("footprint.txt", "w") as of:
        for i in range(len(length_list)):
            length = length_list[i]
            ours_cost = ours_cost_list[i]
            baseline_cost = baseline_cost_list[i]
            ours_new_allocate = ours_acc_new_allocate_list[i]
            baseline_acc_new_allocate = baseline_acc_new_allocate_list[i]
            of.write(f"{length}, {ours_cost}, {baseline_cost}, "
                     f"{ours_new_allocate}, {baseline_acc_new_allocate}\n")
