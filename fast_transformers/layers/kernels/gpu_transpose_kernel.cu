#include "fast_transformers/layers/kernels/gpu_activation_kernel.h"
#include <immintrin.h>
#include <numeric>
#include <cuda_runtime.h>
#include <assert.h>
#include <cstdio>
//#include "fast_transformers/core/cuda_error.h"
//#include "fast_transformers/core/enforce.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

/*
   (batch_size, seq_len, weight_num, num_attention_heads, size_per_head) ->
   (weight_num, batch_size, head_num, seq_len, size_per_head)
   */
template<typename T>
static __global__
void split_add_bias_transpose_for_score(const T* input_data, const T* bias_data, T* output_data,
  const int batch_size, const int seq_len, const int head_num, const int size_per_head, const int word_per_block)
{
  const T* data_ptr;
  T* buf_ptr;
  const T* bias_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int col_offset = blockIdx.x % 3;
  int row_offset = blockIdx.x / 3 * word_per_block % m;

  data_ptr = input_data + row_offset * 3 * n + col_offset * n;
  buf_ptr = output_data + col_offset * m * n;
  bias_ptr = bias_data + col_offset * n;

  int head_id = threadIdx.x % n / size_per_head;
  int hidden_id = threadIdx.x % size_per_head;

  T bias = __ldg(&bias_ptr[threadIdx.x]);

  for(int i = row_offset; i < row_offset + word_per_block; ++i)
  {
    if(i >= m) break;
    int seq_id = i % seq_len;
    int batch_id = i / seq_len;

    int target_id = batch_id * (seq_len * head_num * size_per_head) + head_id * seq_len * size_per_head +
      seq_id * size_per_head + hidden_id;
    T tmp = data_ptr[threadIdx.x] + bias;

    buf_ptr[target_id] = tmp;
    data_ptr += 3 * n;
  }
}

template <typename T>
void GPUSplitAddBiasTransposeForScore(const T* input_data, const T* bias_data, T* out_data,
     int64_t batch_size, int64_t seq_len, int64_t weight_num,
     int64_t num_attention_heads, int64_t size_per_head,
     cudaStream_t stream) {
  const int word_per_block = 32;
  const int n = num_attention_heads * size_per_head;
  const int m = batch_size * seq_len;
  assert(n < 1024);
  //x, y
  dim3 grid(3 * (m+word_per_block-1) / word_per_block);
  dim3 block(n);
  split_add_bias_transpose_for_score<T><<<grid, block, 0, stream>>>(input_data, bias_data, out_data,
      batch_size, seq_len, num_attention_heads, size_per_head, word_per_block);
}

template
void GPUSplitAddBiasTransposeForScore<float>(const float* input_data, const float* bias_data, float* out_data,
     int64_t batch_size, int64_t seq_len, int64_t weight_num,
     int64_t num_attention_heads, int64_t size_per_head,
     cudaStream_t stream);


//copyright nvidia
template<typename T>
static __global__
void transpose(const T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len))/ seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head
    + head_id * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
void GPUTransposeForScore(const T* input_data, T* output_data,
     int64_t batch_size, int64_t seq_len, int64_t num_attention_heads,
     int64_t size_per_head, cudaStream_t stream) {

  const int seq_per_block = 1;
  dim3 grid, block;
  grid.x = batch_size * num_attention_heads
    * seq_len / seq_per_block;
  block.x = seq_per_block * size_per_head;
  transpose<T><<<grid, block, 0, stream>>>(input_data, output_data
      , batch_size, seq_len, num_attention_heads, size_per_head);
}

template
void GPUTransposeForScore<float>(const float* input_data, float* output_data,
     int64_t batch_size, int64_t seq_len, int64_t num_attention_heads,
     int64_t size_per_head, cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
