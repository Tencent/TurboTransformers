if(NOT WITH_GPU)
  return()
endif()

find_package(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(SEND_ERROR "Not found CUDA.")
endif ()

message(STATUS "CUDA detected: " ${CUDA_VERSION})
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=\\\"sm_52,compute_52\\\" -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall -std=c++11 --expt-relaxed-constexpr --expt-extended-lambda")

include_directories(${CUDA_PATH}/include)
link_directories("${CUDA_PATH}/lib64/")

message(STATUS "Fast Transformer is built with GPU.")
