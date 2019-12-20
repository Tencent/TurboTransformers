#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/softmax.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/enforce.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {
namespace kernels {

extern void SoftmaxMask(float* qk_buf, const float* attr_mask,
                        int64_t batch_size, int64_t head_num, int64_t seq_len,
                        float scale);

void SoftmaxMaskNavie(float* qk_buf, const float* attr_mask,
                      const int64_t batch_size, const int64_t head_num,
                      const int64_t seq_len, float scaler) {
  int64_t M = batch_size * head_num * seq_len;
  int64_t N = seq_len;
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float mask_val = attr_mask[i / (head_num * seq_len) * seq_len + j];
      float qk_val = qk_buf[i * N + j];
      qk_val = qk_val * scaler + mask_val;
      qk_buf[i * N + j] = std::exp(qk_val);
    }
    float sum = 0.;
    for (int64_t j = 0; j < N; ++j) {
      sum += qk_buf[i * N + j];
    }
    cblas_sscal(N, 1.0f / (sum + 1e-6), &qk_buf[i * N], 1);
  }  // for i
}

TEST_CASE("softmax_test") {
  int64_t num_attention_heads = 12;

  constexpr float scaler = 1.;
  const int step = 100;

  static core::AlignedScratchpad<float> buf;
  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto attention_scores_size =
          batch_size * num_attention_heads * seq_length * seq_length;

      float* qk_buf = buf.mutable_data(attention_scores_size);
      float* qk_buf_avx = buf.mutable_data(attention_scores_size);
      float* attr_mask = buf.mutable_data(batch_size * num_attention_heads);

      for (int64_t i = 0; i < attention_scores_size; ++i) {
        qk_buf_avx[i] = 0.01;
        qk_buf_avx[i] = 0.01;
      }

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;

      SoftmaxMaskNavie(qk_buf, attr_mask, batch_size, num_attention_heads,
                       seq_length, scaler);

      auto start = std::chrono::system_clock::now();
      for (int i = 0; i < step; ++i) {
        SoftmaxMaskNavie(qk_buf, attr_mask, batch_size, num_attention_heads,
                         seq_length, scaler);
      }
      auto end = std::chrono::system_clock::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      LOG_S(INFO) << "SoftmaxMask Serial cost:"
                  << double(duration.count()) *
                         std::chrono::microseconds::period::num /
                         std::chrono::microseconds::period::den / step * 1000
                  << "ms";

      SoftmaxMask(qk_buf_avx, attr_mask, batch_size, num_attention_heads,
                  seq_length, scaler);

      start = std::chrono::system_clock::now();
      for (int i = 0; i < step; ++i) {
        SoftmaxMask(qk_buf_avx, attr_mask, batch_size, num_attention_heads,
                    seq_length, scaler);
      }
      end = std::chrono::system_clock::system_clock::now();
      auto duration_parallel =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      LOG_S(INFO) << "SoftmaxMask SIMD + Parallel cost:"
                  << double(duration_parallel.count()) *
                         std::chrono::microseconds::period::num /
                         std::chrono::microseconds::period::den / step * 1000
                  << "ms"
                  << ", speedup "
                  << duration.count() * 1. / duration_parallel.count();

      for (int64_t i = 0; i < attention_scores_size; ++i) {
        FT_ENFORCE_LT(fabs(qk_buf[i] - qk_buf_avx[i]), 1e-6, "Wrong @ %d", i);
      }
    }
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
