

#include <chrono>

#include "benchmark_help.h"
#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/softmax.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

static void SoftmaxBenchmarkHelper(int batch_size, int seq_length,
                                   int num_attention_heads, DLDeviceType dev,
                                   int n_step) {
  constexpr float scaler = 1.;
  auto g_bytes = batch_size * num_attention_heads * seq_length * seq_length *
                 sizeof(float) / 1e9;
  core::Tensor qk_buf_tensor(core::NewDLPackTensorT<float>(
      {batch_size, num_attention_heads, seq_length, seq_length}, dev, 0));
  common::FillRandom<float>(qk_buf_tensor);
  core::Tensor attr_mask_tensor(
      core::NewDLPackTensorT<float>({batch_size, seq_length}, dev, 0));
  common::FillRandom<float>(attr_mask_tensor);
  auto res = benchmark::TestFuncSpeed(
      [&]() { ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler); },
      n_step, "", g_bytes, dev);
  std::cout << "GPU Softmax " << batch_size << ", " << seq_length << " " << res
            << " GB/s";
}

TEST_CASE("softmax-cpu-benchmark") {
  constexpr int64_t num_attention_heads = 12;
  constexpr int n_step = 150;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      SoftmaxBenchmarkHelper(batch_size, seq_length, num_attention_heads,
                             kDLCPU, n_step);
    }
}

#ifdef TT_WITH_CUDA
TEST_CASE("softmax-gpu-benchmark") {
  constexpr int64_t num_attention_heads = 12;

  constexpr int n_step = 150;

  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      std::stringstream ss;
      ss << "GPU Softmax " << batch_size << ", " << seq_length << " ";
      SoftmaxBenchmarkHelper(batch_size, seq_length, num_attention_heads,
                             kDLGPU, n_step);
    }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
