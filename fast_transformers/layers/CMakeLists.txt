add_subdirectory(kernels)

add_library(bert_embedding bert_embedding.cpp)
target_link_libraries(bert_embedding glog layer_norm)
