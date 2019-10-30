#include "fast_transformers/ops/cpu_layernorm_op.h"
#include <cassert>
#include <cmath>

namespace fast_transformers {
namespace ops {

template <class T>
void cpu_add_bias(const T* bias,
    T* data,
    const int M,
    const int N) {
    #pragma omp parallel for
    for(int i = 0; i < M; ++i) {
      T bias_val = bias[i];
      T* data_ptr = data + N * i;
      #pragma omp simd
      for(int j = 0; j < N; ++j) {
          //std::cerr << "id " << omp_get_thread_num() << std::endl;
          data_ptr[j] += bias_val;
      }
    }
}

template
void cpu_add_bias<float>(const float* bias,
    float* data,
    const int M,
    const int N); 


template <class T>
void cpu_add_bias_transpose_for_score_long(const T* a, const T* bias,
    const std::vector<unsigned>& shape, T* b)
{
    assert(shape.size() == 4 || shape.size() == 3);
    if(shape.size() == 4) {
      unsigned B = shape[0], S = shape[1], N = shape[2], H = shape[3];
      #pragma omp parallel for
      for(int idx = 0; idx < B * S; ++idx) {
        int i = idx / S;
        int j = idx % S;
        for(int p = 0; p < N; ++p) {
          const T* oldpos = a + i*(S*N*H) + j*N*H + p*H;
          T* newpos = b + i*(S*N*H) + j*H + p*S*H;
          #pragma omp simd
          for(int q = 0; q < H; ++q) {
            newpos[q] = oldpos[q] + bias[p*H + q];
          }
        }
      }
    } else {
      std::cerr << __LINE__ << ", mkl_trans4D error";
    }
}

template
void cpu_add_bias_transpose_for_score_long<float>(const float* a,
    const float* bias,
    const std::vector<unsigned>& shape,
    float* b);

template <class T>
void cpu_transpose_for_score_long(const T* a, const std::vector<unsigned>& shape, T* b)
{
    assert(shape.size() == 4 || shape.size() == 3);
    if(shape.size() == 4) {
      unsigned B = shape[0], S = shape[1], N = shape[2], H = shape[3];
      #pragma omp parallel for
      for(int i = 0; i < B * S; ++i) {
        int s = i % S;
        int batch = i / S;
        for(int p = 0; p < N; ++p) {
          const T* oldpos = a + batch * (S * N * H) + s * N * H + p * H;
          T* newpos = b + batch * (S * N * H) + s * H + p * S * H;
          memcpy(newpos, oldpos, H*sizeof(T));
        }
      }
    } else if (shape.size() == 3) {
        unsigned S = shape[0], N = shape[1], H = shape[2];
        #pragma omp parallel for
        for(int j = 0; j < S; ++j)
          for(int p = 0; p < N; ++p) {
            const T* oldpos = a + j*N*H + p*H;
            T* newpos = b + j*H + p*S*H;
            memcpy(newpos, oldpos, H*sizeof(T));
          }

    } else {
      std::cerr << __LINE__ << ", mkl_trans4D error";
    }
}

template void cpu_transpose_for_score_long<float>(const float* a, const std::vector<unsigned>& shape,float * b);

template <class T>
void cpu_transpose_for_score_short(const T* a, const std::vector<unsigned>& shape, T* b)
{
    assert(shape.size() == 4 || shape.size() == 3);
    if(shape.size() == 4) {
      unsigned B = shape[0], S = shape[1], N = shape[2], H = shape[3];
      for(int i = 0; i < B; ++i)
        for(int j = 0; j < S; ++j)
          for(int p = 0; p < N; ++p) {
            const T* oldpos = a + i*(S*N*H) + j*N*H + p*H;
            T* newpos = b + i*(S*N*H) + j*H + p*S*H;
            memcpy(newpos, oldpos, H*sizeof(T));
          }
    } else if (shape.size() == 3) {
        unsigned S = shape[0], N = shape[1], H = shape[2];
        for(int j = 0; j < S; ++j)
          for(int p = 0; p < N; ++p) {
            const T* oldpos = a + j*N*H + p*H;
            T* newpos = b + j*H + p*S*H;
            memcpy(newpos, oldpos, H*sizeof(T));
          }

    } else {
      std::cerr << __LINE__ << ", mkl_trans4D error";
    }
}

template void cpu_transpose_for_score_short<float>(const float* a, const std::vector<unsigned>& shape, float* b);

/***
 * qk_buf_ (B, Head, seq_len, seq_len)
 * mask_val (B, 1, seq_len, seq_len)
 * **/
/***
 * laynorm((output + bias) + bias)
 * bias (hidden_dim = head_num * size_per_head)
 * ***/
template<typename T>
void cpu_add_bias_input_layernorm_op(T* out, const T* input, const T* bias, 
    const T* gamma, const T* beta, int m, int n) { 
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < m; ++batch_idx) {
            float mean = 0;
            float var = 0;
#pragma omp simd reduction (+:mean)
            for (int i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
                int j = i - batch_idx * n;
                float t = out[i] =  out[i] + input[i] + bias[j];
                mean += t;
                var += t * t;
            }
            mean = mean / n;
            var = var / n - mean * mean;

            // 1 / sqrt(var)
            var = 1.f / sqrtf(var + 1e-6f);

#pragma omp simd
            for (int i = 0; i < n; ++i) {
                int j = batch_idx * n + i;
                out[j] = beta[i] + gamma[i] * var * (out[j] - mean);
            }
        }
}

template void cpu_add_bias_input_layernorm_op<float>(
  float* out, const float* input, const float* bias, const float* gamma, const float* beta, 
  int m, int n);

template<typename T>
void cpu_layernorm_op(T* input, 
    const T* gamma, const T* beta, const int m, const int n) { 
#pragma omp parallel for
        for (int batch_idx = 0; batch_idx < m; ++batch_idx) {
            T mean = 0;
            T var = 0;
#pragma omp simd reduction (+:mean)
            for (int i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
                //int j = i - batch_idx * n;
                T t = input[i];
                mean += t;
                var += t * t;
            }
            mean = mean / n;
            var = var / n - mean * mean;

            // 1 / sqrt(var)
            var = 1.f / sqrtf(var + 1e-6f);

#pragma omp simd
            for (int i = 0; i < n; ++i) {
                int j = batch_idx * n + i;
                input[j] = beta[i] + gamma[i] * var * (input[j] - mean);
            }
        }
}

template void cpu_layernorm_op<float>(
  float* input, const float* gamma, const float* beta, 
  const int m, const int n);


template <typename T>
static T gelu(const T x)
{
  T cdf = 0.5f * (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

} //namespace fast_transformers
} //namespace ops