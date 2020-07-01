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

include(ExternalProject)
add_library(dlpack INTERFACE)
target_include_directories(dlpack INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)
add_subdirectory(pybind11)
set(BUILD_TESTING OFF)
add_subdirectory(abseil)
add_subdirectory(Catch2)
add_library(loguru loguru/loguru.cpp)
target_include_directories(loguru PUBLIC loguru/)
target_compile_definitions(loguru PUBLIC LOGURU_WITH_STREAMS=1)


ExternalProject_Add(extern_zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.11
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a
        )

add_library(zlib STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET zlib PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zlib/include/)
target_include_directories(zlib INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/zlib/include/)
add_dependencies(zlib extern_zlib)

ExternalProject_Add(extern_cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
        GIT_TAG e6536edb128bd0575bf07514d486eb67b5dfb04b
        CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/cnpy
        CMAKE_ARGS -DZLIB_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a
        CMAKE_ARGS -DZLIB_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zlib/include
        CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/cnpy/lib/libcnpy.a
        )

add_dependencies(extern_cnpy extern_zlib)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cnpy/include/)
add_library(cnpy STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET cnpy PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cnpy/lib/libcnpy.a)
target_include_directories(cnpy INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/cnpy/include/)
add_dependencies(cnpy extern_cnpy)

if (${BLAS_PROVIDER} STREQUAL "openblas")
    include(openblas.cmake)
    include(eigen.cmake)
endif()


if (${BLAS_PROVIDER} STREQUAL "blis")
  message(STATUS "CMAKE_CURRENT_BINARY_DIR " ${CMAKE_CURRENT_BINARY_DIR})
        ExternalProject_Add(extern_blis
                GIT_REPOSITORY https://github.com/flame/blis.git
                GIT_TAG 0.7.0
                BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/blis
                SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/blis
                INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/blis
                CONFIGURE_COMMAND COMMAND ""
                BUILD_COMMAND COMMAND ./configure --enable-threading=openmp --enable-cblas --prefix=${CMAKE_CURRENT_BINARY_DIR}/blis auto && make -j ${nproc} && make check -j ${nproc}
                INSTALL_COMMAND COMMAND make install
                BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/blis/lib/libblis.a)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/blis/include/blis/)
        add_library(blis STATIC IMPORTED GLOBAL)
        SET_PROPERTY(TARGET blis PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/blis/lib/libblis.a)
        target_include_directories(blis INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/blis/include/blis/)
        add_dependencies(blis extern_blis)
        include(eigen.cmake)
endif()
