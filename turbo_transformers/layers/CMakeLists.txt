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

add_subdirectory(kernels)

add_library(tt_layers OBJECT
        bert_embedding.cpp
        bert_attention.cpp
        bert_intermediate.cpp
        bert_output.cpp
        sequence_pool.cpp
        bert_pooler.cpp
        prepare_bert_masks.cpp
        albert_layer.cpp
        multi_headed_attention.cpp
        multi_headed_attention_smart_batch.cpp
        positionwise_ffn.cpp
        addbias_act.cpp
        addbias_layernorm.cpp
        )

target_link_libraries(tt_layers PUBLIC tt_core tt_kernels)

add_executable(tt_layers_test prepare_bert_masks_test.cpp)
target_link_libraries(tt_layers_test catch2_test_main tt_layers tt_core tt_kernels)
