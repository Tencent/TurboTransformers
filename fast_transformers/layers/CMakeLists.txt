add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding loguru layer_norm)

add_library(bert_attention bert_attention.cpp)
#target_include_directories(bert_attention PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(bert_attention tensor blas loguru enforce softmax transpose)

add_library(bert_intermediate bert_intermediate.cpp)
target_link_libraries(bert_intermediate tensor blas loguru aligned_scratchpad Eigen3::Eigen)

add_library(bert_output bert_output.cpp)
target_link_libraries(bert_output tensor blas loguru enforce activation)


add_library(sequence_pool sequence_pool.cpp)
target_link_libraries(sequence_pool tensor seq_pool)
