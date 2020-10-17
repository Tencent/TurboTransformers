
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

add_library(catch2_test_main OBJECT catch2_test_main.cpp)
target_link_libraries(catch2_test_main Catch2)
add_library(tt_core
        OBJECT
            blas_${BLAS_PROVIDER}.cpp
            enforce.cpp
            memory.cpp
            tensor.cpp
            config.cpp
            profiler.cpp
            allocator/allocator_impl.cpp
            allocator/bert_config.cpp
        allocator/model_aware_memory_scheduler.cpp
            allocator/ordered_list.cpp
            allocator/base_allocator.cpp
            allocator/naive_allocator.cpp
            allocator/model_aware_allocator.cpp
        allocator/allocator_api.cpp
        )
target_link_libraries(tt_core PUBLIC
        absl::stacktrace
        absl::symbolize
        absl::strings
        absl::str_format
        absl::variant
        dlpack
        loguru
        ${RT_LIBRARY}
        ${CMAKE_DL_LIBS}
        )
if (${BLAS_PROVIDER} STREQUAL "mkl")
    target_link_libraries(tt_core PUBLIC ${MKL_LIBRARIES})
    target_include_directories(tt_core PUBLIC ${MKL_INCLUDE_DIR})
    target_compile_definitions(tt_core PUBLIC -DTT_BLAS_USE_MKL)
elseif (${BLAS_PROVIDER} STREQUAL "openblas")
    target_link_libraries(tt_core PUBLIC OpenBlas::OpenBlas PUBLIC Eigen3::Eigen)
    target_compile_definitions(tt_core PUBLIC -DTT_BLAS_USE_OPENBLAS)
elseif (${BLAS_PROVIDER} STREQUAL "blis")
    target_link_libraries(tt_core PUBLIC blis Eigen3::Eigen)
    target_compile_definitions(tt_core PUBLIC -DTT_BLAS_USE_BLIS)
endif ()


if (WITH_GPU)
    target_sources(tt_core PRIVATE cuda_device_context.cpp)
    target_link_libraries(tt_core PUBLIC cudart cuda cublas)
endif()

add_executable(tt_core_test
        enforce_test.cpp
        device_context_test.cpp
        tensor_test.cpp
        allocator/allocator_api_test.cpp
        allocator/bert_allocator_test.cpp
        allocator/model_aware_memory_scheduler_test.cpp
        fp16_test.cpp)
target_link_libraries(tt_core_test catch2_test_main tt_core)
add_test(NAME tt_core_test  COMMAND tt_core_test)
