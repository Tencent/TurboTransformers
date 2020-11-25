import numpy as np
import math


def greedy_by_size_offset_calculation(usage_recorders, show_detail=False):
    """
    Allocator memory on a continous memory space
    @params
        input: usage_recorders (name, start_op, end_op, size)
        show_detail: show debug info
    @return
        offset on the memory space for each tensor (name, offset)
    """
    usage_recorders = sorted(usage_recorders,
                             key=lambda tup: tup[3],
                             reverse=True)
    recorders_size = len(usage_recorders)
    offset = {}

    total_consumption = 0
    # name, first_op, last, size, offset
    ordered_allocated_ids = []

    # TODO(jiaruifang) O(N) which is very time-consuming for 12-layer bert,
    # which consists of too many operators.
    # However, we should only calculate offsets for one layer.
    # The other layers should reuse the same space.
    # rewrite this function into C++ code may help.
    for t in usage_recorders:
        t_name = t[0]
        t_size = t[3]
        prev_offset = 0
        best_offset = None
        smallest_gap = math.inf
        for x in ordered_allocated_ids:
            x_size = x[3]
            x_idx = x[0]
            x_offset = x[4]
            max_first_op = max(t[1], x[1])
            min_last_op = min(t[2], x[2])
            if max_first_op <= min_last_op:
                gap = x_offset - prev_offset
                if gap >= t_size and gap < smallest_gap:
                    smallest_gap = gap
                    best_offset = prev_offset
                prev_offset = max(prev_offset, x_offset + x_size)

        if best_offset is None:
            best_offset = prev_offset
        offset[t_name] = best_offset
        total_consumption = max(total_consumption, best_offset + t_size)
        t = (*t, best_offset)
        ordered_allocated_ids.append(t)
        #TODO(jiaruifang) time consuming part
        ordered_allocated_ids = sorted(ordered_allocated_ids,
                                       key=lambda elem: elem[4])

    if show_detail:
        print("tensor_id, offset")
        for item in offset.items():
            print(item[0], item[1])
    total_consumption = total_consumption / 1024 / 1024
    print(
        f"greedy_by_size_offset_calculation footprint {total_consumption} MB")

    return offset, total_consumption


if __name__ == "__main__":
    from bert_tensor_usage import get_bert_tensor_usage_record
    tur = get_bert_tensor_usage_record(20, 512)
    # print(tur)
    greedy_by_size_offset_calculation(tur, True)

    tur = get_bert_tensor_usage_record(1, 20)
    # print(tur)
    greedy_by_size_offset_calculation(tur, True)
