cmake_minimum_required(VERSION 3.14)
project(fast_transformers)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

option(CUDA_VERSION "Set cuda version of torch, in (`92`, `101`), default is None" NONE)
option(OPENMP_ENDABLE "Build OpenMP" ON)

add_subdirectory(3rd)
enable_testing()


if (OPENMP_ENDABLE)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        message(STATUS "OpenMP FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        message(WARNING "OpenMP is not supported by the compiler")
    endif ()
else ()
    message(STATUS "OpenMP is not used")
endif ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(fast_transformers)
