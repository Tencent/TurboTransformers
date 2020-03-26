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

include(ExternalProject)
set(OPENBLAS_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR}/openblas/)
ExternalProject_Add(extern_openblas
        GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
        GIT_TAG v0.3.9
        INSTALL_DIR ${OPENBLAS_BIN_ROOT}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${OPENBLAS_BIN_ROOT}
        CMAKE_ARGS -DBUILD_WITHOUT_LAPACK=ON
        BUILD_BYPRODUCTS ${OPENBLAS_BIN_ROOT}/lib/libopenblas.a
        )

add_library(openblas_openblas STATIC IMPORTED GLOBAL)
add_dependencies(openblas_openblas extern_openblas)
file(MAKE_DIRECTORY ${OPENBLAS_BIN_ROOT}/include/openblas)
SET_PROPERTY(TARGET openblas_openblas PROPERTY IMPORTED_LOCATION ${OPENBLAS_BIN_ROOT}/lib/libopenblas.a)
target_include_directories(openblas_openblas INTERFACE ${OPENBLAS_BIN_ROOT}/include/openblas)


add_library(OpenBlas::OpenBlas ALIAS openblas_openblas)
