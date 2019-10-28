include(ExternalProject)
add_library(dlpack INTERFACE)
target_include_directories(dlpack INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)
add_subdirectory(pybind11)

#IF (NOT UNIX)
#    MESSAGE(FATAL "Only linux and macos are supported")
#ENDIF ()
#
#
#IF (APPLE)
#    SET(LIBTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.3.0.zip)
#    SET(LIBTORCH_MD5 180651185e3e10c1d386dfeeb099e25f)
#    SET(LIBTORCH_FILE libtorch.dylib)
#ELSE ()
#    SET(LIBTORCH_FILE libtorch.so)
#    IF (CUDA_VERSION STREQUAL "92")
#        SET(LIBTORCH_URL
#                https://download.pytorch.org/libtorch/cu92/libtorch-cxx11-abi-shared-with-deps-1.3.0%2Bcu92.zip)
#        SET(LIBTORCH_MD5 5a965a29bf9f3e1538641e1a6eec1187)
#    ELSEIF (CUDA_VERSION STREQUAL "101")
#        SET(LIBTORCH_URL
#                https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.0.zip)
#        SET(LIBTORCH_MD5 05f1a14771af5fd82e303de3e196e7cb)
#    ELSE ()
#        SET(LIBTORCH_URL
#                https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.3.0%2Bcpu.zip)
#        SET(LIBTORCH_MD5 dfb63dc5d142628e21a2ac5aee013255)
#    ENDIF ()
#ENDIF ()
#
#
#ExternalProject_Add(extern_libtorch
#        URL ${LIBTORCH_URL}
#        URL_HASH "MD5=${LIBTORCH_MD5}"
#        CONFIGURE_COMMAND ""
#        BUILD_COMMAND ""
#        INSTALL_COMMAND "")
#SET(LIBTORCH_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern_libtorch-prefix/src/extern_libtorch/)
#add_library(libtorch_shared IMPORTED SHARED GLOBAL)
#set_property(TARGET libtorch_shared PROPERTY IMPORTED_LOCATION
#        "${LIBTORCH_DIR}/lib/${LIBTORCH_FILE}")
#target_include_directories(libtorch_shared INTERFACE ${LIBTORCH_DIR}/include)
#add_dependencies(libtorch_shared extern_libtorch)
