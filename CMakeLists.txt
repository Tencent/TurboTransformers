cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0079 NEW)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(${CMAKE_CURRENT_SOURCE_DIR}/hunter-gate/cmake/HunterGate.cmake)

HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.23.240.tar.gz"
        SHA1 "ca19f3769e6c80cfdd19d8b12ba5102c27b074e0"
)

option(WITH_PROFILER "Compile with gperftools" OFF)
option(WITH_GPU "Build with GPU" OFF)

if (WITH_GPU)
  project(fast_transformers LANGUAGES CXX C CUDA)
else ()
  project(fast_transformers LANGUAGES CXX C)
endif ()

hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

if(WITH_GPU)
  set(CUDA_PATH "/data1/qspace/kanowang/common/cuda-9.0" CACHE PATH "The cuda library root")
  include(cuda)
  add_definitions(-DFT_WITH_CUDA)
endif()

set(MKLROOT "/opt/intel/mkl" CACHE PATH "The mkl library root")
set(BLAS_PROVIDER "mkl" CACHE STRING "Set the blas provider library, in [openblas, mkl]")
set(FAST_TRANSFORMERS_VERSION 0.1.1)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL Intel)
    message(STATUS "Fast Transformer is built with a intel compiler!")
    add_definitions(-D__USE_INTEL_COMPILER__)
endif ()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_C_FLAGS "-Wall")

add_subdirectory(3rd)
enable_testing()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP USED FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "OpenMP is not supported by the compiler")
endif ()

message(STATUS "Blas provider is ${BLAS_PROVIDER}")

if (${BLAS_PROVIDER} STREQUAL "mkl")
    find_package(MKL REQUIRED)
    add_definitions(
            -DEIGEN_USE_THREADS # support openmp
            -DEIGEN_USE_MKL_ALL # support mkl
            -DEIGEN_NO_MALLOC  # if any eigen code needs malloc, just raise a compilation error
            -DEIGEN_DEFAULT_TO_ROW_MAJOR
            -DEIGEN_FAST_MATH=1 # fast math for cos/sin/...
    )


elseif (${BLAS_PROVIDER} STREQUAL "openblas")
    hunter_add_package(OpenBLAS)
    find_package(OpenBLAS CONFIG REQUIRED)

    add_definitions(
            -DEIGEN_USE_THREADS # support openmp
            -DEIGEN_NO_MALLOC  # if any eigen code needs malloc, just raise a compilation error
            -DEIGEN_DEFAULT_TO_ROW_MAJOR
            -DEIGEN_FAST_MATH=1 # fast math for cos/sin/...
    )

else ()
    message(SEND_ERROR "Blas Provider must in (openblas, mkl)")
endif ()

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
    target_link_libraries(absl_base PRIVATE ${RT_LIBRARY})
ENDIF ()


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(fast_transformers)
