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

add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding loguru layer_norm)
if (WITH_GPU)
  target_link_libraries(bert_embedding gpu_embedding_kernel)
endif ()

add_library(bert_attention bert_attention.cpp)
target_link_libraries(bert_attention tensor blas loguru enforce softmax transpose aligned_scratchpad)

add_library(bert_intermediate bert_intermediate.cpp)
target_link_libraries(bert_intermediate loguru activation mat_mul)

add_library(bert_output bert_output.cpp)
target_link_libraries(bert_output loguru activation mat_mul)

add_library(sequence_pool sequence_pool.cpp)
target_link_libraries(sequence_pool tensor seq_pool)

add_library(prepare_bert_masks prepare_bert_masks.cpp)
target_link_libraries(prepare_bert_masks tensor common)

if (WITH_GPU)
  target_link_libraries(prepare_bert_masks gpu_utils)
endif ()

add_executable(prepare_bert_masks_test prepare_bert_masks_test.cpp)
target_link_libraries(prepare_bert_masks_test prepare_bert_masks Catch2 loguru enforce aligned_scratchpad)
if (WITH_GPU)
  target_link_libraries(prepare_bert_masks_test enforce cuda_device_context memory)
endif()
