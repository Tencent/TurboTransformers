

#pragma once
#include <stdint.h>

#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T, layers::types::PoolType t>
void GPUReduceAxisOne(const T* input, T* output, int batch_size, int seq_len,
                      int hidden_size);

template <typename T>
void GPUSequence(T* data_ptr, int64_t size);

template <typename T>
void GPUFill(T* data_ptr, int64_t size, T val);

extern void GPUTransform(int64_t* src_data_ptr, float* dst_data_ptr,
                         const int64_t size);
template <bool AddInput, typename T>
void GPUAddBias(const T* input1, const T* input2, const T* bias, int64_t m,
                int64_t n, cudaStream_t stream, T* out);

template <typename Dtype>
void GPUConcat(const Dtype* t1, const Dtype* t2, const int64_t high_dim,
               const int64_t t1_mid_size, const int64_t t2_mid_size,
               const int64_t low_dim, cudaStream_t stream, Dtype* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
