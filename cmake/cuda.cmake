if (NOT WITH_GPU)
  return()
endif ()

set(gpu_archs9 "35 50 52 60 61 70")
set(gpu_archs10 "35 50 52 60 61 70 75")

if (NOT DEFINED ENV{CUDA_PATH})
  find_package(CUDA REQUIRED)
  if (CUDA_FOUND)
    message(STATUS "CUDA detected: " ${CUDA_VERSION})
  else ()
    message(SEND_ERROR "Not defined CUDA_PATH and Not found CUDA.")
  endif ()
endif ()

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
include_directories(${CUDA_PATH}/include)
link_directories("${CUDA_PATH}/lib64/")

if (${CUDA_VERSION} LESS 10.0)
  set(gpu_archs ${gpu_archs9})
elseif (${CUDA_VERSION} LESS 11.0)
  set(gpu_archs ${gpu_archs10})
else ()
  message(SEND_ERROR "This CUDA_VERSION is not support now.")
endif ()

set(cuda_arch_bin ${gpu_archs})
string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
list(REMOVE_DUPLICATES cuda_arch_bin)
list(GET cuda_arch_bin -1 cuda_arch_ptx)
list(REMOVE_AT cuda_arch_bin -1)

set(nvcc_flags "")
foreach(arch ${cuda_arch_bin})
  set(nvcc_flags "${nvcc_flags} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

set(nvcc_flags "${nvcc_flags} -gencode arch=compute_${cuda_arch_ptx},code=\\\"sm_${cuda_arch_ptx},compute_${cuda_arch_ptx}\\\"")
message(STATUS ${nvcc_flags})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${nvcc_flags} -rdc=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -std=c++11 --expt-relaxed-constexpr --expt-extended-lambda")
