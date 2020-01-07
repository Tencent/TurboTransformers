#include <cuda_runtime.h>
#include <immintrin.h>
#include <cassert>
#include <numeric>
#include "fast_transformers/layers/kernels/gpu_embedding_kernel.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

static __global__ void lookup(float* dst, const float* embedding_table,
                              const int64_t* ids, int64_t vocab_size) {
  int64_t id = ids[blockIdx.x];
  int hidden_idx = threadIdx.x;
  int hidden_size = blockDim.x;
  assert(id < vocab_size);

  float val = __ldg(&embedding_table[id * hidden_size + hidden_idx]);
  dst[blockIdx.x * hidden_size + hidden_idx] = val;
}

static __global__ void add_lookup(float* dst, const float* embedding_table,
                                  const int64_t* ids, int64_t vocab_size) {
  int64_t id = ids[blockIdx.x];
  int hidden_idx = threadIdx.x;
  int hidden_size = blockDim.x;
  assert(id < vocab_size);

  float val = __ldg(&embedding_table[id * hidden_size + hidden_idx]);
  dst[blockIdx.x * hidden_size + hidden_idx] += val;
}

void GPULookupKernel(float* dst, const float* embedding_table,
                     const int64_t* ids, int64_t vocab_size,
                     int64_t hidden_size, int64_t num_ids, bool Add,
                     cudaStream_t stream) {
  dim3 grid(num_ids);
  dim3 block(hidden_size);
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPULookupKernel thread block size large than 1024");
  }
  if (Add) {
    add_lookup<<<grid, block, 0, stream>>>(dst, embedding_table, ids,
                                           vocab_size);
  } else {
    lookup<<<grid, block, 0, stream>>>(dst, embedding_table, ids, vocab_size);
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
