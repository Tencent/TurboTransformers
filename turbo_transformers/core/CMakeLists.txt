
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

add_library(blas blas_${BLAS_PROVIDER}.cpp)

if (${BLAS_PROVIDER} STREQUAL "mkl")
    target_link_libraries(blas PRIVATE ${MKL_LIBRARIES})
    target_include_directories(blas PUBLIC ${MKL_INCLUDE_DIR})
    target_compile_definitions(blas PUBLIC -DFT_BLAS_USE_MKL)
elseif (${BLAS_PROVIDER} STREQUAL "openblas")
    target_link_libraries(blas PUBLIC OpenBlas::OpenBlas PRIVATE Eigen3::Eigen)
    target_compile_definitions(blas PUBLIC -DFT_BLAS_USE_OPENBLAS)
endif ()

target_link_libraries(blas PRIVATE ${CMAKE_DL_LIBS})

add_library(common common.cpp)
target_link_libraries(common tensor dlpack)
set(CUDA_DEPS)
if (WITH_GPU)
    target_link_libraries(common gpu_utils)
    set(CUDA_DEPS cudart cuda cublas)
endif ()

add_library(enforce enforce.cpp)
target_link_libraries(enforce PUBLIC ${CUDA_DEPS} absl::stacktrace absl::symbolize absl::strings absl::str_format PRIVATE ${RT_LIBRARY})

add_executable(enforce_test enforce_test.cpp)
target_link_libraries(enforce_test Catch2 enforce)
add_test(NAME enforce_test COMMAND enforce_test)

add_library(memory memory.cpp)
target_link_libraries(memory enforce dlpack)

add_library(tensor tensor.cpp)
target_link_libraries(tensor enforce dlpack blas memory absl::variant)

if (WITH_GPU)
    add_library(cuda_device_context cuda_device_context.cpp)
    target_link_libraries(cuda_device_context enforce tensor)

    add_library(cuda_allocator cuda_allocator.cpp)
    target_link_libraries(cuda_allocator enforce)

    add_executable(device_context_test device_context_test.cpp)
    target_link_libraries(device_context_test cuda_device_context Catch2 tensor)

    target_link_libraries(tensor cuda_allocator cuda_device_context)
endif ()

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test Catch2 tensor)
add_test(NAME tensor_test COMMAND tensor_test)

add_library(profiler profiler.cpp)
set(PROFILER_DEPS loguru enforce)
if (WITH_PROFILER)
    list(APPEND PROFILER_DEPS gperftools::profiler)
endif ()
target_link_libraries(profiler ${PROFILER_DEPS})

add_library(aligned_scratchpad aligned_scratchpad.cpp ../loaders/modeling_bert.h)
target_link_libraries(aligned_scratchpad PRIVATE memory enforce)
add_executable(aligned_scratchpad_test aligned_scratchpad_test.cpp)
target_link_libraries(aligned_scratchpad_test aligned_scratchpad Catch2)
add_test(NAME aligned_scratchpad_test COMMAND aligned_scratchpad_test)

add_executable(fp16_test fp16_test.cpp)
target_link_libraries(fp16_test Catch2)

add_library(tensor_copy INTERFACE)
target_link_libraries(tensor_copy INTERFACE tensor memory)
