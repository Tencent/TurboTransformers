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

cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_C_FLAGS "-Wall")


set(TURBO_TRANSFORMERS_VERSION 0.6.0)


option(WITH_PROFILER  "Compile with profiler"   OFF)
option(WITH_GPU       "Build with GPU"          OFF)
option(WITH_MODULE_BENCHMAKR       "Catch2 unitest with benchmarking"          ON)
option(WITH_TENSOR_CORE       "Use Tensor core to accelerate"          OFF)


if (WITH_GPU)
  project(turbo_transformers LANGUAGES CXX C CUDA)
else ()
  project(turbo_transformers LANGUAGES CXX C)
endif ()

if(WITH_GPU)
  set(CUDA_PATH "/usr/local/cuda" CACHE PATH "The cuda library root")
  include(cuda)
  add_definitions(-DTT_WITH_CUDA)
  include_directories(3rd/cub)
endif()

set(MKLROOT "/opt/intel/mkl" CACHE PATH "The mkl library root")
set(BLAS_PROVIDER "mkl" CACHE STRING "Set the blas provider library, in [openblas, mkl, blis]")
if (${BLAS_PROVIDER} STREQUAL "mkl")
    find_package(MKL REQUIRED)
endif()

message(STATUS "Blas provider is ${BLAS_PROVIDER}")

add_subdirectory(3rd)
include_directories(3rd/FP16/include)
enable_testing()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP USED FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "OpenMP is not supported")
endif ()


if (WITH_PROFILER)
    add_definitions(-DWITH_PERFTOOLS)
endif ()

if (WITH_GPU AND WITH_TENSOR_CORE)
    add_definitions(-DWITH_TENSOR_CORE)
endif ()

IF (UNIX AND NOT APPLE)
    # Link absl_base needs -lrt on linux. It is necessary on CentOS.
    find_library(RT_LIBRARY NAMES librt.a
            PATHS $ENV{CONDA_PREFIX}/lib/
            $ENV{CONDA_PREFIX}/x86_64-conda_cos6-linux-gnu/sysroot/lib/
            $ENV{CONDA_PREFIX}/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib/)
    if (NOT RT_LIBRARY)
        message(SEND_ERROR "Cannot find librt from")
    ENDIF ()
ENDIF ()
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(turbo_transformers)
if (WITH_MODULE_BENCHMAKR)
    add_definitions(-DCATCH_CONFIG_ENABLE_BENCHMARKING)
    add_subdirectory(benchmark)
endif()
add_subdirectory(example)
