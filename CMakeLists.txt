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

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_C_FLAGS "-Wall")

set(turbo_transformers_VERSION 0.1.1)

option(WITH_PROFILER  "Compile with gperftools" OFF)
option(WITH_GPU       "Build with GPU"          OFF)

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
set(BLAS_PROVIDER "mkl" CACHE STRING "Set the blas provider library, in [openblas, mkl]")
if (${BLAS_PROVIDER} STREQUAL "mkl")
    find_package(MKL REQUIRED)
endif()
if (${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    message(STATUS "Fast Transformer is built with a intel compiler!")
    add_definitions(-D__USE_INTEL_COMPILER__)
endif ()

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

message(STATUS "Blas provider is ${BLAS_PROVIDER}")

if (WITH_PROFILER)
    find_package(Gperftools REQUIRED)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
    add_definitions(-DWITH_GPERFTOOLS)
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
