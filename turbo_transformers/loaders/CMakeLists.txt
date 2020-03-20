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

add_library(modeling_bert modeling_bert.cpp)
target_link_libraries(modeling_bert
        PUBLIC dlpack
        PRIVATE cnpy zlib tensor_copy bert_embedding bert_attention bert_intermediate bert_output prepare_bert_masks
        sequence_pool)

add_executable(modeling_bert_test modeling_bert_test.cpp)
target_link_libraries(modeling_bert_test modeling_bert Catch2)
add_test(NAME modeling_bert_test COMMAND modeling_bert_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
