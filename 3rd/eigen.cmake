

include(ExternalProject)
set(EIGEN_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR}/eigen/)
ExternalProject_Add(extern_eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.3.7
        INSTALL_DIR ${EIGEN_BIN_ROOT}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EIGEN_BIN_ROOT}
        BUILD_BYPRODUCTS ${EIGEN_BIN_ROOT}/include
        )

add_library(eigen_eigen INTERFACE)
add_dependencies(eigen_eigen extern_eigen)
target_include_directories(eigen_eigen INTERFACE ${EIGEN_BIN_ROOT}/include/eigen3)
target_compile_definitions(eigen_eigen INTERFACE
        -DEIGEN_USE_THREADS # support openmp
        -DEIGEN_NO_MALLOC  # if any eigen code needs malloc, just raise a compilation error
        -DEIGEN_DEFAULT_TO_ROW_MAJOR
        -DEIGEN_FAST_MATH=1 # fast math for cos/sin/...
        )
add_library(Eigen3::Eigen ALIAS eigen_eigen)
