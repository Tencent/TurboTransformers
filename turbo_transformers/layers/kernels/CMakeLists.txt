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

add_library(tt_kernels OBJECT
        layer_norm.cpp softmax.cpp transpose.cpp activation.cpp
        common.cpp seq_pool.cpp mat_mul.cpp)
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
        softmax_test.cpp
        transpose_test.cpp
        layer_norm_test.cpp
        mat_mul_test.cpp)

target_link_libraries(tt_kernels_test tt_kernels tt_core catch2_test_main)
add_test(NAME tt_kernels_test COMMAND tt_kernels_test)
