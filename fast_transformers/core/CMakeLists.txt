add_library(enforce enforce.cpp)
target_link_libraries(enforce PUBLIC absl::stacktrace absl::symbolize absl::strings absl::str_format)

add_executable(enforce_test enforce_test.cpp)
target_link_libraries(enforce_test Catch2 enforce)
add_test(NAME enforce_test COMMAND enforce_test)

add_library(memory memory.cpp)
target_link_libraries(memory enforce)

add_library(tensor tensor.cpp)
target_link_libraries(tensor dlpack blas memory)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test Catch2 tensor)
add_test(NAME tensor_test COMMAND tensor_test)

add_library(profiler profiler.cpp)
target_link_libraries(profiler glog gperftools::profiler enforce)

add_library(blas blas.cpp)
target_link_libraries(blas PRIVATE absl::strings ghc_filesystem dl
        PUBLIC absl::variant enforce)

add_executable(blas_test blas_test.cpp)
target_link_libraries(blas_test blas Catch2)
add_test(NAME blas_test COMMAND blas_test)

add_library(aligned_scratchpad aligned_scratchpad.cpp)
target_link_libraries(aligned_scratchpad PRIVATE memory enforce)
add_executable(aligned_scratchpad_test aligned_scratchpad_test.cpp)
target_link_libraries(aligned_scratchpad_test aligned_scratchpad Catch2)
