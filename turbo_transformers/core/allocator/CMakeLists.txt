
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


add_library(tt_model_aware_allocator
        bert_allocator_config.cpp
        chunked_allocator.cpp)
target_link_libraries(tt_model_aware_allocator PUBLIC
        loguru
        tt_core
        ${RT_LIBRARY}
        ${CMAKE_DL_LIBS}
        )

add_executable(bert_allocator_config_test
        bert_allocator_config_test.cpp)
target_link_libraries(bert_allocator_config_test tt_model_aware_allocator catch2_test_main)
add_test(NAME bert_allocator_config_test  COMMAND bert_allocator_config_test)

add_executable(chunked_allocator_test
        chunked_allocator_test.cpp)
target_link_libraries(chunked_allocator_test tt_model_aware_allocator catch2_test_main)
add_test(NAME chunked_allocator_test  COMMAND chunked_allocator_test)
