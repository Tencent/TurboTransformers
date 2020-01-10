add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding loguru layer_norm)
if (WITH_GPU)
  target_link_libraries(bert_embedding gpu_embedding_kernel)
endif ()

add_library(bert_attention bert_attention.cpp)
target_link_libraries(bert_attention tensor blas loguru enforce softmax transpose aligned_scratchpad)

add_library(bert_intermediate bert_intermediate.cpp)
target_link_libraries(bert_intermediate loguru activation mat_mul)

add_library(bert_output bert_output.cpp)
target_link_libraries(bert_output loguru activation mat_mul)


add_library(sequence_pool sequence_pool.cpp)
target_link_libraries(sequence_pool tensor seq_pool)

add_library(prepare_bert_masks prepare_bert_masks.cpp)
target_link_libraries(prepare_bert_masks eigen-tensor)
