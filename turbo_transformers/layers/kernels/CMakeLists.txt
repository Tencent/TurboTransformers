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

add_library(tt_kernels OBJECT
        layer_norm.cpp softmax.cpp transpose.cpp activation.cpp
        common.cpp seq_pool.cpp mat_mul.cpp embedding.cpp utils.cpp)
target_link_libraries(tt_kernels PUBLIC tt_core)

if (WITH_GPU)
    target_sources(tt_kernels PRIVATE
            gpu_layer_norm_kernel.cu
            gpu_softmax_kernel.cu
            gpu_activation_kernel.cu
            gpu_transpose_kernel.cu
            gpu_embedding_kernel.cu
            gpu_utils.cu
            )
    target_link_libraries(tt_kernels PUBLIC cudart cuda)
endif()

add_executable(tt_kernels_test
        activation_test.cpp
        embedding_test.cpp
        softmax_test.cpp
        transpose_test.cpp
        layer_norm_test.cpp
        mat_mul_test.cpp
        utils_test.cpp
        gpu_utils_test.cpp)

target_link_libraries(tt_kernels_test tt_kernels tt_core catch2_test_main)
add_test(NAME tt_kernels_test COMMAND tt_kernels_test)
