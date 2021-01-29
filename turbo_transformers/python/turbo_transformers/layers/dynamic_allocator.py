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
import math
import time
DEFAULT_TRUNK_SIZE = 2 * 1024 * 1024
K_SCALE = 1.2


class Trunk:
    def __init__(self, size=DEFAULT_TRUNK_SIZE):
        self._size = size
        # a list of tensor (name, first_op, last_op, size, offset)
        self._tensor_list = []


class TrunkList:
    def __init__(self):
        self._trunks = []

    def appendTrunk(self, trunk):
        self._trunks.append(trunk)

    def getInfo(self):
        info = []
        for trunk in self._trunks:
            info.append(trunk._size)
        return info


gTrunkList = TrunkList()


# trunk is a list of ordered_allocated_ids
def try_fit_trunk(t, trunk: Trunk):
    """
    @params
      t : tensor, which is a tuple (name, first_op, last_op, size)
      trunk : a Trunk
    @returns
      trunk : an updated list of tuples (name, first_op, last_op, size, offset)
      best_offset : None or a offset
    """
    # first_op, last, size, id, offset

    t_size = t[3]
    prev_offset = 0
    best_offset = None
    smallest_gap = math.inf
    for x in trunk._tensor_list:
        x_size = x[3]
        x_offset = x[4]
        max_first_op = max(t[1], x[1])
        min_last_op = min(t[2], x[2])
        if max_first_op <= min_last_op:
            gap = x_offset - prev_offset
            if gap >= t_size and gap < smallest_gap:
                smallest_gap = gap
                best_offset = prev_offset
            prev_offset = max(prev_offset, x_offset + x_size)

    # the left space of this trunk is enought for this tensor
    if best_offset is None and trunk._size - prev_offset >= t_size:
        best_offset = prev_offset

    if best_offset is None:
        return trunk, None
    else:
        t = (*t, best_offset)
        trunk._tensor_list.append(t)
        # sort tensor info list according to their offsets
        trunk._tensor_list = sorted(trunk._tensor_list,
                                    key=lambda elem: elem[4])
        return trunk, best_offset


def trunked_greedy_by_size_offset_calculation(usage_recorders,
                                              show_detail=False):
    """
    An offset calculation algorithm designed for variable-length inputs.
    @ params:
    usage_recorders : tensor usage recoders (name, start_op, end_op, size)
    global trunk_size_list : a list of list (name, offset)
    @returns:
    assigned_offset : a dict indicates the offset for each tensor
    assigned_trunk : a dict indicates the trunk for each tensor
    """
    global gTrunkList
    # descend
    usage_recorders = sorted(usage_recorders,
                             key=lambda tup: tup[3],
                             reverse=False)
    assigned_offset = {}
    assigned_trunk = {}
    new_allocate_size = 0

    time_start = time.time()
    for i in range(len(gTrunkList._trunks)):
        gTrunkList._trunks[i]._tensor_list = []

    for t in usage_recorders:
        t_name = t[0]
        t_size = t[3]
        is_assigned = False
        for trunk_id, trunk in enumerate(gTrunkList._trunks):
            trunk, offset = try_fit_trunk(t, trunk)
            if offset is not None:
                assigned_trunk[t_name] = trunk_id
                assigned_offset[t_name] = offset
                # update gTrunkList
                gTrunkList._trunks[trunk_id] = trunk
                is_assigned = True
                break

        # init new trunk, trunk id should be assigned after delete useless trunk
        if is_assigned is False:
            trunk_size = max(DEFAULT_TRUNK_SIZE,
                             math.ceil((t_size * K_SCALE + 31) // 32 * 32))
            new_allocate_size += trunk_size
            trunk = Trunk(trunk_size)
            trunk._tensor_list.append((*t, 0))  #offset @ 0
            gTrunkList.appendTrunk(trunk)

            # TODO
            trunk_id = len(gTrunkList._trunks) - 1
            assigned_trunk[t_name] = trunk_id
            assigned_offset[t_name] = 0

    time_end = time.time()
    core_cost = time_end - time_start

    used_consumption = 0
    total_consumption = 0
    delete_trunk_list = []

    # find trunk not used -> delete_trunk_list
    for trunk_id, trunk in enumerate(gTrunkList._trunks):
        max_end_offset = 0
        for elem in trunk._tensor_list:
            max_end_offset = max(elem[4] + elem[3],
                                 max_end_offset)  # offset + size
        # print("trunk id", trunk_id, " usage ",
        #       max_end_offset / gTrunkList._trunks[trunk_id]._size)
        used_consumption += max_end_offset
        if max_end_offset == 0:
            delete_trunk_list.insert(0, trunk_id)
        else:
            total_consumption += gTrunkList._trunks[trunk_id]._size

    # delete
    for id in delete_trunk_list:
        gTrunkList._trunks.pop(id)

    # adjust trunk ids
    for trunk_id, trunk in enumerate(gTrunkList._trunks):
        for tensor in trunk._tensor_list:
            tensor_name = tensor[0]
            assigned_trunk[tensor_name] = trunk_id

    if show_detail:
        print("=====allocation plan====")
        print("trunk_id \t size")
        for i, t in enumerate(gTrunkList._trunks):
            print(i, t._size)
        print("tensor name \t offset")
        for t in assigned_offset.items():
            t_name = t[0]
            print("{", t_name, assigned_trunk[t_name], assigned_offset[t_name],
                  "},")
            # print("{\"" + t_name + "\",", assigned_offset[t_name], "},")
        print("=====allocation plan====")

    used_consumption = used_consumption / 1024 / 1024
    total_consumption = total_consumption / 1024 / 1024
    new_allocate_size = new_allocate_size / 1024 / 1024
    if show_detail:
        print(
            f"> debug total_consumption {total_consumption} MB used_consumption {used_consumption} MB percent {used_consumption/total_consumption}"
        )
    return assigned_offset, assigned_trunk, gTrunkList.getInfo(), (
        total_consumption, new_allocate_size)


if __name__ == "__main__":
    from bert_tensor_usage import get_bert_tensor_usage_record

    # for length in [200, 240]:
    #     print(f"begin schedule for allocate {length}")
    #     tur = get_bert_tensor_usage_record(1, length, 1)
    #     print(tur[4])
    #     trunked_greedy_by_size_offset_calculation(tur, True)
    #     print("\n\n")
    tur = get_bert_tensor_usage_record(1, 25, 12)
    print(tur)
    for t in tur:
        print("{\"" + f"{t[0]}" + "\",", f"{t[1]}, {t[2]}, {t[3]}", "},")

    trunked_greedy_by_size_offset_calculation(tur, True)
