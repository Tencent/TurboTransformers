cmake_minimum_required(VERSION 3.14)
project(fast_transformers)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
option(CUDA_VERSION "Set cuda version of torch, in (`92`, `101`), default is None" NONE)
add_subdirectory(3rd)
enable_testing()
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(fast_transformers)

add_executable(test_fast_transformer 
    ./unitest/test_tensor.cpp
    ./unitest/unitest.cpp
    )
target_link_libraries(test_fast_transformer tensor gtest)

enable_testing()
message(STATUS "test includes: ${googletest_INCLUDE_DIRS} lib: ${googletest_STATIC_LIBRARIES}")
target_include_directories(test_fast_transformer PUBLIC ./fast_transformers)