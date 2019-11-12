set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -mavx2 -mfma") #-mavx512f

add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding glog layer_norm)

add_library(attention bert_attention.cpp)
#target_include_directories(attention PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(attention tensor blas glog enforce softmax transpose)

add_library(bert_self_attention bert_self_attention.cpp)
#target_include_directories(bert_self_attention PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(bert_self_attention tensor blas glog enforce softmax transpose)

add_library(bert_intermediate bert_intermediate.cpp)
target_link_libraries(bert_intermediate tensor blas glog aligned_scratchpad)

add_library(bert_output bert_output.cpp)
target_link_libraries(bert_output tensor blas glog enforce activation)
