add_library(layer_norm layer_norm.cpp)
target_link_libraries(layer_norm tensor)

add_library(softmax softmax.cpp)
target_link_libraries(softmax tensor)

add_library(transpose transpose.cpp)
target_link_libraries(transpose tensor)

add_library(activation activation.cpp)
target_link_libraries(activation tensor aligned_scratchpad)

if (WITH_GPU)
  add_library(gpu_softmax_kernel gpu_softmax_kernel.cu)
  target_link_libraries(gpu_softmax_kernel cudart cuda)
  target_link_libraries(softmax gpu_softmax_kernel)

  add_library(gpu_activation_kernel gpu_activation_kernel.cu)
  target_link_libraries(gpu_activation_kernel cudart cuda)
  target_link_libraries(activation gpu_activation_kernel)

  add_library(gpu_transpose_kernel gpu_transpose_kernel.cu)
  target_link_libraries(gpu_transpose_kernel cudart cuda)
  target_link_libraries(transpose gpu_transpose_kernel)
endif()


add_executable(activation_test activation_test.cpp)
target_link_libraries(activation_test activation Catch2 loguru enforce aligned_scratchpad)
if (WITH_GPU)
  target_link_libraries(activation_test cuda_error cuda_device_context memory)
endif()

add_executable(softmax_test softmax_test.cpp)
target_link_libraries(softmax_test softmax Catch2 loguru enforce aligned_scratchpad)
if (WITH_GPU)
  target_link_libraries(softmax_test cuda_error cuda_device_context memory)
endif()

add_executable(transpose_test transpose_test.cpp)
target_link_libraries(transpose_test transpose Catch2 loguru enforce aligned_scratchpad)
if (WITH_GPU)
  target_link_libraries(transpose_test cuda_error cuda_device_context memory)
endif()


add_library(seq_pool seq_pool.cpp)
target_link_libraries(seq_pool tensor enforce)

add_library(mat_mul mat_mul.cpp)
target_link_libraries(mat_mul tensor enforce)
if (WITH_GPU)
  target_link_libraries(mat_mul cuda_device_context)
endif()

add_executable(mat_mul_test mat_mul_test.cpp)
target_link_libraries(mat_mul_test blas Catch2 mat_mul tensor)
