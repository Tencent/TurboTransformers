#define CATCH_CONFIG_MAIN
#include "fast_transformers/layers/kernels/activation.h"
#include <chrono>
#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/enforce.h"
#include "loguru.hpp"

#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_error.h"
#include "fast_transformers/core/memory.h"
#endif

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

using Tensor = fast_transformers::core::Tensor;

#ifdef FT_WITH_CUDA
template <typename T>
inline void FillCPUGPU(Tensor& cpu_tensor, Tensor& gpu_tensor) {
  T* gpu_data = gpu_tensor.mutableData<T>();
  T* cpu_data = cpu_tensor.mutableData<T>();
  auto size = cpu_tensor.numel();
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  fast_transformers::core::FT_Memcpy(
      gpu_data, cpu_data, size * sizeof(T),
      ::fast_transformers::core::MemcpyFlag::kCPU2GPU);
}

template <typename T>
inline bool CompareCPUGPU(const Tensor& cpu_tensor, const Tensor& gpu_tensor) {
  const T* gpu_data = gpu_tensor.data<T>();
  const T* cpu_data = cpu_tensor.data<T>();
  auto size = cpu_tensor.numel();

  std::unique_ptr<T[]> gpu_data_ref(new T[size]);
  fast_transformers::core::FT_Memcpy(
      gpu_data_ref.get(), gpu_data, size,
      fast_transformers::core::MemcpyFlag::kGPU2CPU);
  bool ret = true;
  for (int64_t i = 0; i < size; ++i) {
    if (std::fab(gpu_data_ref[i] - cpu_data[i]) > 1e-3) {
      ret = false;
      break;
    }
  }
  return ret;
}

TEST_CASE("CPU and GPU result correctness") {
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

      FillCPUGPU<float>(cpu_bias, gpu_bias);
      FillCPUGPU<float>(cpu_out, gpu_out);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        AddBiasGeLUAct<float>(cpu_bias, &cpu_out);
        AddBiasGeLUAct<float>(gpu_bias, &gpu_out);
      }
      REQUIRE(CompareCPUGPU<float>(cpu_out, gpu_out));
    }  // for
}
#endif

TEST_CASE("activation_cpu_test") {
  int64_t hidden_size = 12 * 64;

  const int step = 100;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      fast_transformers::core::Tensor bias(nullptr);
      bias.Reshape<float>({n});
      fast_transformers::core::Tensor out(nullptr);
      out.Reshape<float>({m, n});
      fast_transformers::core::Tensor out_parallel(nullptr);
      out_parallel.Reshape<float>({m, n});

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
template <typename T>
inline void Fill(fast_transformers::core::Tensor& tensor) {
  T* gpu_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T> cpu_data(new T[size]);
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  fast_transformers::core::FT_Memcpy(
      gpu_data, cpu_data.get(), size * sizeof(T),
      fast_transformers::core::MemcpyFlag::kCPU2GPU);
}

TEST_CASE("activation_gpu_test") {
  int64_t hidden_size = 12 * 64;

  const int step = 100;

  std::vector<int64_t> batch_size_list{1, 20, 24};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};

  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      auto m = batch_size * seq_length;
      auto n = hidden_size;

      fast_transformers::core::Tensor bias(
          fast_transformers::core::NewDLPackTensorT<float>({hidden_size},
                                                           kDLGPU, 0));
      Fill<float>(bias);

      fast_transformers::core::Tensor out(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length, hidden_size}, kDLGPU, 0));
      Fill<float>(out);

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
