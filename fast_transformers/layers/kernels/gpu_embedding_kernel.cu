#include <cuda_runtime.h>
#include <immintrin.h>
#include <numeric>
#include "fast_transformers/layers/kernels/gpu_embedding_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

template <bool IsAdd>
static __global__ void lookup(float* dst, const float* embedding_table,
                              const int64_t* ids, int64_t vocab_size) {
  int64_t id = ids[blockIdx.x];
  int hidden_idx = threadIdx.x;
  int hidden_size = blockDim.x;
  // assert(id < vocab_size);

  float val = __ldg(&embedding_table[id * hidden_size + hidden_idx]);
  if (IsAdd) {
    dst[blockIdx.x * hidden_size + hidden_idx] += val;
  } else {
    dst[blockIdx.x * hidden_size + hidden_idx] = val;
  }
}

void GPULookupKernel(float* dst, const float* embedding_table,
                     const int64_t* ids, int64_t vocab_size,
                     int64_t hidden_size, int64_t num_ids, bool is_add,
                     cudaStream_t stream) {
  dim3 grid(num_ids);
  dim3 block(hidden_size);
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPULookupKernel currently dose not support a hidden_size than 1024");
  }
  add_lookup<<<grid, block, 0, stream>>>
      <is_add>(dst, embedding_table, ids, vocab_size);
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
