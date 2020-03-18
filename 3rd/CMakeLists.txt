include(ExternalProject)
add_library(dlpack INTERFACE)
target_include_directories(dlpack INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)
add_subdirectory(pybind11)
set(BUILD_TESTING OFF)
add_subdirectory(abseil)
add_subdirectory(Catch2)
add_library(loguru loguru/loguru.cpp)
target_include_directories(loguru PUBLIC loguru/)
target_compile_definitions(loguru PUBLIC LOGURU_WITH_STREAMS=1)


ExternalProject_Add(extern_zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.2.11
        INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a
        )

add_library(zlib STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET zlib PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zlib/include/)
target_include_directories(zlib INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/zlib/include/)
add_dependencies(zlib extern_zlib)

ExternalProject_Add(extern_cnpy
        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
        GIT_TAG 4e8810b1a8637695171ed346ce68f6984e585ef4
        CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/zlib/
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/cnpy
        CMAKE_ARGS -DZLIB_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/zlib/lib/libz.a
        CMAKE_ARGS -DZLIB_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/zlib/include
        CMAKE_ARGS -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/cnpy/lib/libcnpy.a
        )

add_dependencies(extern_cnpy extern_zlib)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cnpy/include/)
add_library(cnpy STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET cnpy PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cnpy/lib/libcnpy.a)
target_include_directories(cnpy INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/cnpy/include/)
add_dependencies(cnpy extern_cnpy)
