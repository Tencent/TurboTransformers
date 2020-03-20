add_library(modeling_bert modeling_bert.cpp)
target_link_libraries(modeling_bert
        PUBLIC dlpack
        PRIVATE cnpy zlib tensor_copy bert_embedding bert_attention bert_intermediate bert_output prepare_bert_masks
        sequence_pool)

add_executable(modeling_bert_test modeling_bert_test.cpp)
target_link_libraries(modeling_bert_test modeling_bert Catch2)
add_test(NAME modeling_bert_test COMMAND modeling_bert_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
