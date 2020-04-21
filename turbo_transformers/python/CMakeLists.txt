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

pybind11_add_module(turbo_transformers_cxx pybind.cpp)

target_link_libraries(turbo_transformers_cxx PRIVATE
        tt_core tt_layers tt_kernels)

SET(PY_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pypackage)
file(GLOB_RECURSE PY_PROJ_FILES ${CMAKE_CURRENT_SOURCE_DIR}/turbo_transformers/*.py)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${PY_PACKAGE_DIR}/setup.py @ONLY)

add_custom_command(OUTPUT ${PY_PACKAGE_DIR}/.timestamp
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/turbo_transformers ${PY_PACKAGE_DIR}/turbo_transformers
        COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:turbo_transformers_cxx>"
        "${PY_PACKAGE_DIR}/turbo_transformers/$<TARGET_FILE_NAME:turbo_transformers_cxx>"
        COMMAND env CC=${CMAKE_C_COMPILER} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E touch ${PY_PACKAGE_DIR}/.timestamp
        WORKING_DIRECTORY ${PY_PACKAGE_DIR}
        DEPENDS turbo_transformers_cxx ${PY_PROJ_FILES})
add_custom_target(turbo_transformers ALL DEPENDS ${PY_PACKAGE_DIR}/.timestamp)

file(GLOB PY_TEST_FILES tests/*_test.py)

foreach (py_test_file ${PY_TEST_FILES})
    get_filename_component(py_test_trg_name ${py_test_file} NAME_WE)
    message(STATUS "Add python test ${py_test_file}")
    add_test(NAME
            ${py_test_trg_name}
            COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PY_PACKAGE_DIR}
            ${PYTHON_EXECUTABLE} ${py_test_file}
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()
