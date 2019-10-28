cmake_minimum_required(VERSION 3.14)
project(fast_transformers)

set(CMAKE_CXX_STANDARD 14)
option(CUDA_VERSION "Set cuda version of torch, in (`92`, `101`), default is None" NONE)
enable_testing()
add_subdirectory(3rd)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(fast_transformers)

