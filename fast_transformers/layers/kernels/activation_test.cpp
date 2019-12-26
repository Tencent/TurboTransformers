#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/activation.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {
namespace kernels {

void AddBiasGeLUActNaive(const float* bias, float* out, int64_t m, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    int64_t k = 0;
    for (int64_t j = n * i; j < n * (i + 1); ++j) {
      auto before_act = out[j] + bias[k++];
      out[j] = before_act * 0.5f *
               (1.0f + std::tanh(0.7978845608028654f *
                                 (before_act + 0.044715f * before_act *
                                                   before_act * before_act)));
    }
  }
}
TEST_CASE("activation CPU benchmark") {
  int64_t hidden_size = 12 * 64;

  const int step = 10;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      fast_transformers::core::Tensor bias(nullptr);
      bias.Reshape<float>({n}, kDLCPU, 0);
      fast_transformers::core::Tensor out(nullptr);
      out.Reshape<float>({m, n}, kDLCPU, 0);
      fast_transformers::core::Tensor out_parallel(nullptr);
      out_parallel.Reshape<float>({m, n}, kDLCPU, 0);

      float* bias_ptr = bias.mutableData<float>();
      float* out_ptr = out.mutableData<float>();
      float* out_parallel_ptr = out_parallel.mutableData<float>();

      for (int64_t i = 0; i < n; ++i) {
        bias_ptr[i] = 0.01;
      }
      for (int64_t i = 0; i < m * n; ++i) {
        out_ptr[i] = out_parallel_ptr[i] = 0.02;
      }

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;

      {
        AddBiasGeLUActNaive(bias.data<float>(),
                            out_parallel.mutableData<float>(), m, n);

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < step; ++i) {
          AddBiasGeLUActNaive(bias.data<float>(),
                              out_parallel.mutableData<float>(), m, n);
        }
        auto end = std::chrono::system_clock::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto elapse = double(duration.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den / step * 1000;
        LOG_S(INFO) << "AddBiasGeLUActNaive cost:" << elapse
                    << " ms, Bandwidth " << m * n * 0.1 / 1e6 / elapse
                    << " GB/s";
      }
      {
        AddBiasGeLUAct<float>(bias, &out);

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < step; ++i) {
          AddBiasGeLUAct<float>(bias, &out);
        }
        auto end = std::chrono::system_clock::system_clock::now();
        auto duration_parallel =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto elapse = double(duration_parallel.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den / step * 1000;
        LOG_S(INFO) << "AddBiasGeLUAct OMP cost:" << elapse << " ms, Bandwidth "
                    << m * n * 0.1 / 1e6 / elapse << " GB/s";
      }
      for (int64_t i = 0; i < m * n; ++i) {
        FT_ENFORCE_LT(fabs(out_parallel_ptr[i] - out_parallel_ptr[i]), 1e-6,
                      "Wrong @ %d", i);
      }
    }
}

#ifdef FT_WITH_CUDA
TEST_CASE("activation CPU and GPU correctness") {
  int64_t hidden_size = 12 * 64;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor gpu_bias(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLGPU, 0));

      fast_transformers::core::Tensor cpu_bias(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLCPU, 0));

      fast_transformers::core::Tensor gpu_out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));

      fast_transformers::core::Tensor cpu_out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLCPU, 0));

      ::fast_transformers::test::FillCPUGPU<float>(cpu_bias, gpu_bias);
      ::fast_transformers::test::FillCPUGPU<float>(cpu_out, gpu_out);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        AddBiasGeLUAct<float>(cpu_bias, &cpu_out);
        AddBiasGeLUAct<float>(gpu_bias, &gpu_out);
      }
      REQUIRE(
          ::fast_transformers::test::CompareCPUGPU<float>(cpu_out, gpu_out));
    }  // for
}

TEST_CASE("activation GPU benchmark") {
  int64_t hidden_size = 12 * 64;

  const int step = 10;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      fast_transformers::core::Tensor bias(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLGPU, 0));
      ::fast_transformers::test::Fill<float>(bias);

      fast_transformers::core::Tensor out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));
      ::fast_transformers::test::Fill<float>(out);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        AddBiasGeLUAct<float>(bias, &out);

        auto start = std::chrono::system_clock::now();
        for (int i = 0; i < step; ++i) {
          AddBiasGeLUAct<float>(bias, &out);
        }
        auto end = std::chrono::system_clock::system_clock::now();
        auto duration_parallel =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto elapse = double(duration_parallel.count()) *
                      std::chrono::microseconds::period::num /
                      std::chrono::microseconds::period::den / step * 1000;
        LOG_S(INFO) << "AddBiasGeLUAct GPU cost:" << elapse << " ms, Bandwidth "
                    << m * n * 0.1 / 1e6 / elapse << " GB/s";
      }
    }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
