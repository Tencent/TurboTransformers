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

__all__ = ['get_bert_tensor_usage_record']


def get_bert_tensor_usage_record(batch_size,
                                 seq_len,
                                 num_head=12,
                                 hidden_size=768,
                                 num_layer=12):
    """
    build a tensor usage recorders for bert.
    Inputs : parameters to define a BERT.
    Outputs : tensor usage records (TURs) (name, start_op, end_op, size)
    """
    data_bytes = 4
    start_idx = 0
    from_seq_len = seq_len
    to_seq_len = seq_len

    Q_size = batch_size * from_seq_len * hidden_size * data_bytes
    K_size = V_size = batch_size * to_seq_len * hidden_size * data_bytes
    attn_score_size = batch_size * num_head * from_seq_len * to_seq_len * data_bytes
    TUR_dict = {}
    TUR_dict[f"qkv_out1"] = (start_idx + 0, start_idx + 1,
                             K_size + Q_size + V_size)
    TUR_dict[f"q"] = (start_idx + 1, start_idx + 2, Q_size)
    TUR_dict[f"k"] = (start_idx + 1, start_idx + 2, K_size)
    TUR_dict[f"v"] = (start_idx + 1, start_idx + 3, V_size)
    TUR_dict[f"attn_score"] = (start_idx + 2, start_idx + 3, attn_score_size)
    TUR_dict[f"context_layer"] = (start_idx + 3, start_idx + 4, Q_size)
    TUR_dict[f"self_attr_out"] = (start_idx + 4, start_idx + 5,
                                  attn_score_size)
    TUR_dict[f"attn_output"] = (start_idx + 5, start_idx + 8, Q_size)
    TUR_dict[f"intermediate_output"] = (start_idx + 7, start_idx + 8,
                                        Q_size * 4)
    TUR_dict[f"layer_output"] = (start_idx + 0, start_idx + 9, Q_size)

    TUR_list = []
    # print("tensor_name\t start_op, end_op, size")
    for item in TUR_dict.items():
        TUR_list.append((item[0], *item[1]))
    return TUR_list


def get_bert_tensor_usage_record_classic(batch_size,
                                         seq_len,
                                         num_head=12,
                                         hidden_size=768,
                                         num_layer=12):
    """
    build a tensor usage recorders for bert.
    Inputs : parameters to define a BERT.
    Outputs : tensor usage records (TURs) (name, start_op, end_op, size)
    """
    data_bytes = 4
    start_idx = 0
    from_seq_len = seq_len
    to_seq_len = seq_len

    Q_size = batch_size * from_seq_len * hidden_size * data_bytes
    K_size = V_size = batch_size * to_seq_len * hidden_size * data_bytes
    attn_score_size = batch_size * num_head * from_seq_len * to_seq_len * data_bytes
    TUR_dict = {}
    for layer_idx in range(num_layer):
        TUR_dict[f"{layer_idx}_qkv_out1"] = (start_idx + 0, start_idx + 1,
                                             K_size + Q_size + V_size)
        TUR_dict[f"{layer_idx}_q"] = (start_idx + 1, start_idx + 2, Q_size)
        TUR_dict[f"{layer_idx}_k"] = (start_idx + 1, start_idx + 2, K_size)
        TUR_dict[f"{layer_idx}_v"] = (start_idx + 1, start_idx + 3, V_size)
        TUR_dict[f"{layer_idx}_attn_score"] = (start_idx + 2, start_idx + 3,
                                               attn_score_size)
        TUR_dict[f"{layer_idx}_context_layer"] = (start_idx + 3, start_idx + 4,
                                                  Q_size)
        TUR_dict[f"{layer_idx}_self_attr_out"] = (start_idx + 4, start_idx + 5,
                                                  attn_score_size)
        TUR_dict[f"{layer_idx}_attn_output"] = (start_idx + 5, start_idx + 8,
                                                Q_size)
        TUR_dict[f"{layer_idx}_intermediate_output"] = (start_idx + 7,
                                                        start_idx + 8,
                                                        Q_size * 4)
        if layer_idx == num_layer - 1:
            TUR_dict[f"{layer_idx}_layer_output"] = (start_idx + 8,
                                                     start_idx + 8, Q_size)
        else:
            TUR_dict[f"{layer_idx}_layer_output"] = (start_idx + 8,
                                                     start_idx + 15, Q_size)

        start_idx += 9
    TUR_list = []
    # print("tensor_name\t start_op, end_op, size")
    for item in TUR_dict.items():
        TUR_list.append((item[0], *item[1]))
    return TUR_list
