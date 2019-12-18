add_library(layer_norm layer_norm.cpp)
target_link_libraries(layer_norm tensor)

add_library(softmax softmax.cpp)
target_link_libraries(softmax tensor)

add_library(transpose transpose.cpp)
target_link_libraries(transpose tensor)

add_library(activation activation.cpp)
target_link_libraries(activation eigen-tensor tensor aligned_scratchpad)
if(FT_WITH_GPU)
add_library(gpu_activation_kernel gpu_activation_kernel.cu)
target_link_libraries(gpu_activation_kernel -lcudart -lcuda)
target_link_libraries(activation gpu_activation_kernel)
endif()


add_executable(activation_test activation_test.cpp)
target_link_libraries(activation_test activation Catch2 loguru enforce aligned_scratchpad device_context)
if(FT_WITH_GPU)
target_link_libraries(activation_test cuda_error)
endif()

add_library(seq_pool seq_pool.cpp)
target_link_libraries(seq_pool tensor enforce)

add_library(mat_mul mat_mul.cpp)
target_link_libraries(mat_mul tensor enforce)
