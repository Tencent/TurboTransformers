add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding loguru layer_norm)

add_library(attention bert_attention.cpp)
#target_include_directories(attention PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(attention tensor blas loguru enforce softmax transpose)

add_library(bert_self_attention bert_self_attention.cpp)
#target_include_directories(bert_self_attention PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(bert_self_attention tensor blas loguru enforce softmax transpose)

add_library(bert_intermediate bert_intermediate.cpp)
target_link_libraries(bert_intermediate tensor blas loguru aligned_scratchpad)

add_library(bert_output bert_output.cpp)
target_link_libraries(bert_output tensor blas loguru enforce activation)
