if (NOT WITH_GPU)
  return()
endif ()

if (NOT DEFINED ENV{CUDA_PATH})
  find_package(CUDA REQUIRED)
  if (NOT CUDA_FOUND)
    message(SEND_ERROR "Not defined CUDA_PATH and Not found CUDA.")
  endif ()
  message(STATUS "CUDA detected: " ${CUDA_VERSION})
endif ()

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
include_directories(${CUDA_PATH}/include)
link_directories("${CUDA_PATH}/lib64/")

# CUDA_ARCH_LIST also can be "Common", CUDA architecture(s) is 3.0;3.5;5.0;5.2;6.0;6.1;7.0;7.0+PTX
if(NOT CUDA_ARCH_LIST)
  set(CUDA_ARCH_LIST "Auto")
endif()

cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
foreach(X ${ARCH_FLAGS})
  set(CUDA_FLAGS "${CUDA_FLAGS} ${X}")
endforeach()

message(STATUS "Generating CUDA code for ${CUDA_VERSION} SMs: ${CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS} -Xcompiler -Wall --expt-relaxed-constexpr --use_fast_math --expt-extended-lambda")
