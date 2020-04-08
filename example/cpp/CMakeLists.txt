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

add_library(bert_model bert_model.cpp)
target_link_libraries(bert_model
        PUBLIC tt_npz_loader
        PRIVATE tt_layers tt_kernels)

add_executable(bert_model_example bert_model_example.cpp)
target_link_libraries(bert_model_example bert_model)

add_executable(tt_cxx_bert_model_test bert_model_test.cpp)
target_link_libraries(tt_cxx_bert_model_test catch2_test_main bert_model)
add_test(NAME tt_cxx_bert_model_test COMMAND tt_cxx_bert_model_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

ADD_CUSTOM_TARGET(copy_bertnpz ALL COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_CURRENT_BINARY_DIR}/models)
