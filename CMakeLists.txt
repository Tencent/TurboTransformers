cmake_minimum_required(VERSION 3.14)
project(fast_transformers)

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(3rd)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(fast_transformers)
