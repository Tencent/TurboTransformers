add_library(blas blas.cpp openblas.cpp)
target_link_libraries(blas PRIVATE absl::strings
        PUBLIC  absl::variant)
