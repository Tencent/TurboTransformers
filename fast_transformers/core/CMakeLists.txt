add_library(blas blas_${BLAS_PROVIDER}.cpp)

if (${BLAS_PROVIDER} STREQUAL "mkl")
    target_link_libraries(blas PRIVATE ${MKL_LIBRARIES})
    target_include_directories(blas PUBLIC ${MKL_INCLUDE_DIR})
    target_compile_definitions(blas PUBLIC -DFT_BLAS_PROVIDER=MKL)
endif ()


target_link_libraries(blas PRIVATE ${CMAKE_DL_LIBS})

add_executable(blas_test blas_test.cpp)
target_link_libraries(blas_test blas Catch2 mat_mul tensor)
add_test(NAME blas_test COMMAND blas_test)

add_library(enforce enforce.cpp)
target_link_libraries(enforce PUBLIC absl::stacktrace absl::symbolize absl::strings absl::str_format)

add_executable(enforce_test enforce_test.cpp)
target_link_libraries(enforce_test Catch2 enforce)
add_test(NAME enforce_test COMMAND enforce_test)

add_library(memory memory.cpp)
target_link_libraries(memory enforce)

add_library(tensor tensor.cpp)
target_link_libraries(tensor dlpack blas memory absl::variant)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test Catch2 tensor)
add_test(NAME tensor_test COMMAND tensor_test)

if (WITH_GPU)
  add_library(cuda_error cuda_error.cpp)
  target_link_libraries(cuda_error enforce cudart cuda cublas)

  add_library(cuda_device_context cuda_device_context.cpp)
  target_link_libraries(cuda_device_context cuda_error enforce tensor)

  add_executable(device_context_test device_context_test.cpp)
  target_link_libraries(device_context_test cuda_device_context Catch2 tensor)
  target_link_libraries(tensor cuda_error)
  target_link_libraries(memory cuda_error)
endif ()


add_library(profiler profiler.cpp)
set(PROFILER_DEPS loguru enforce)
if (WITH_PROFILER)
    list(APPEND PROFILER_DEPS gperftools::profiler)
endif ()
target_link_libraries(profiler ${PROFILER_DEPS})


add_library(aligned_scratchpad aligned_scratchpad.cpp)
target_link_libraries(aligned_scratchpad PRIVATE memory enforce)
add_executable(aligned_scratchpad_test aligned_scratchpad_test.cpp)
target_link_libraries(aligned_scratchpad_test aligned_scratchpad Catch2)

add_library(eigen-tensor eigen-tensor.cpp)
target_link_libraries(eigen-tensor PUBLIC tensor Eigen3::Eigen)
