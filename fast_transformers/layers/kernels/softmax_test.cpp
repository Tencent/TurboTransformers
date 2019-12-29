#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/softmax.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {
namespace kernels {

inline void _CreateBenchmark(DLDeviceType device_type) {
  const std::string device_name = device_type == kDLCPU ? "CPU" : "GPU";
  const int step = 100;

  int64_t num_attention_heads = 12;
  constexpr float scaler = 1.;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor qk_buf_tensor(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length},
              device_type, 0));
      ::fast_transformers::test::Fill<float>(qk_buf_tensor);
      fast_transformers::core::Tensor attr_mask_tensor(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, device_type, 0));
      ::fast_transformers::test::Fill<float>(attr_mask_tensor);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);
      auto start = std::chrono::system_clock::now();
      for (int i = 0; i < step; ++i) {
        ApplyMaskAndSoftmax(&qk_buf_tensor, attr_mask_tensor, scaler);
      }

#ifdef FT_WITH_CUDA
      if (device_type == kDLGPU) cudaDeviceSynchronize();
#endif
      auto end = std::chrono::system_clock::system_clock::now();
      auto duration_parallel =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

      auto elapse = double(duration_parallel.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den / step * 1000;

      auto data_size =
          batch_size * num_attention_heads * seq_length * seq_length;

      LOG_S(INFO) << device_name
                  << ": SoftmaxMask cost:" << data_size / 1e6 / elapse
                  << "GB/s";
    }
}

#ifdef FT_WITH_CUDA
TEST_CASE("softmax CPU and GPU correctness") {
  int64_t num_attention_heads = 12;

  constexpr float scaler = 1.;

  static core::AlignedScratchpad<float> buf;
  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor qk_buf_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length}, kDLGPU,
              0));

      fast_transformers::core::Tensor qk_buf_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, num_attention_heads, seq_length, seq_length}, kDLCPU,
              0));
      ::fast_transformers::test::FillCPUGPU<float>(qk_buf_cpu, qk_buf_gpu);

      fast_transformers::core::Tensor attr_mask_gpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLGPU, 0));

      fast_transformers::core::Tensor attr_mask_cpu(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLCPU, 0));

      ::fast_transformers::test::FillCPUGPU<float>(attr_mask_cpu,
                                                   attr_mask_gpu);

      ApplyMaskAndSoftmax(&qk_buf_gpu, attr_mask_gpu, scaler);

      ApplyMaskAndSoftmax(&qk_buf_cpu, attr_mask_cpu, scaler);

      REQUIRE(::fast_transformers::test::CompareCPUGPU<float>(qk_buf_cpu,
                                                              qk_buf_gpu));
    }
}

TEST_CASE("softmax GPU benchmark") { _CreateBenchmark(kDLGPU); }
#endif

TEST_CASE("softmax CPU benchmark") { _CreateBenchmark(kDLCPU); }

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
