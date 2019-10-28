add_library(enforce enforce.cpp)
target_link_libraries(enforce PUBLIC absl::stacktrace absl::symbolize absl::strings absl::str_format)

add_executable(enforce_test enforce_test.cpp)
target_link_libraries(enforce_test Catch2 enforce)

add_library(tensor tensor.cpp)
target_link_libraries(tensor dlpack enforce)

add_library(blas blas.cpp)
target_link_libraries(blas PRIVATE absl::strings
        PUBLIC absl::variant enforce)

