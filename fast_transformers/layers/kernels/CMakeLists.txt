add_library(layer_norm layer_norm.cpp)
target_link_libraries(layer_norm tensor)

add_library(softmax softmax.cpp)
target_link_libraries(softmax tensor)

add_executable(softmax_test softmax_test.cpp)
target_link_libraries(softmax_test PUBLIC softmax blas Catch2 loguru enforce aligned_scratchpad)

add_library(transpose transpose.cpp)
target_link_libraries(transpose tensor)

add_library(activation activation.cpp)
target_link_libraries(activation eigen-tensor)

add_library(seq_pool seq_pool.cpp)
target_link_libraries(seq_pool tensor enforce)
