#include "fast_transformers/core/memory.h"

#ifdef WITH_CUDA
#include "fast_transformers/core/nvcommon.h"
#endif

namespace fast_transformers {
namespace core {
void *align_alloc(size_t sz, size_t align) {
  void *aligned_mem;
  FT_ENFORCE_EQ(posix_memalign(&aligned_mem, align, sz), 0,
                "Cannot allocate align memory with %d bytes, "
                "align %d",
                sz, align);
  return aligned_mem;
}

#ifdef WITH_CUDA
void *cuda_alloc(size_t sz) {
  void* device_mem;
  check_cuda_error(cudaMalloc((void**)&(device_mem), sz));
  return device_mem;
}

void *cuda_free(void* data) {
  cudaFree(data);
}

template<typename T>
void FT_Memcpy(T* dst_data, const T* src_data, int64_t length, FT_MemcpyFlag flag) {
  if (length <= 0)
	return;
  auto data_size = length * sizeof(T);
  if (flag == FT_GPU2CPU) {
	cudaMemcpy(((void*)dst_data), ((void*)src_data), 
          data_size, cudaMemcpyDeviceToHost);
  } else if (flag == FT_CPU2GPU) {
    cudaMemcpy(((void*)dst_data), ((void*)src_data), 
          data_size, cudaMemcpyHostToDevice);
  } else if (flag == FT_CPU2CPU) {
	std::copy(src_data, src_data + length, dst_data);
  }
}

template
void FT_Memcpy<float>(float* dst_data, const float* src_data, int64_t length, FT_MemcpyFlag flag);
#endif

}  // namespace core
}  // namespace fast_transformers
