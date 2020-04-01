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

add_library(tt_cxx_interface modeling_bert.cpp)
target_link_libraries(tt_cxx_interface
        PUBLIC dlpack
        PRIVATE cnpy zlib tt_core tt_layers tt_kernels)

add_executable(tt_cxx_interface_test modeling_bert_test.cpp)
target_link_libraries(tt_cxx_interface_test catch2_test_main tt_cxx_interface)
add_test(NAME tt_cxx_interface_test COMMAND tt_cxx_interface_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

ADD_CUSTOM_TARGET(bertnpzmodel ALL COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_CURRENT_BINARY_DIR}/models)
