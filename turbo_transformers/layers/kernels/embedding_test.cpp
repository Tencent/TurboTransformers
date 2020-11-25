

#include "turbo_transformers/layers/kernels/embedding.h"

#include "catch2/catch.hpp"
#include "loguru.hpp"
#include "turbo_transformers/layers/kernels/common.h"

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

#ifdef TT_WITH_CUDA
TEST_CASE("embedding-gpu-test") {
  std::vector<int64_t> hidden_size_list{128, 768};
  std::vector<int64_t> vocab_size_list{2000, 5000, 10000, 20000};
  std::vector<int64_t> ids_length_list{10,  20,  40,  60,  80,
                                       100, 200, 300, 400, 500};
  for (auto hidden_size : hidden_size_list)
    for (auto vocab_size : vocab_size_list)
      for (auto ids_length : ids_length_list) {
        core::Tensor gpu_ids(nullptr), cpu_ids(nullptr),
            gpu_embedding_table(nullptr), cpu_embedding_table(nullptr),
            gpu_out(nullptr), cpu_out(nullptr);
        cpu_ids = common::CreateTensorAndFillConstant<int64_t>({ids_length},
                                                               kDLCPU, 0, 10);
        gpu_ids = common::CreateTensorAndFillConstant<int64_t>({ids_length},
                                                               kDLGPU, 0, 10);
        std::tie(cpu_embedding_table, gpu_embedding_table) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {vocab_size, hidden_size});
        std::tie(cpu_out, gpu_out) =
            common::CreateAndFillRandomForCPUGPUTensors<float>(
                {ids_length, hidden_size});

        {
          LookupEmbedding<true>(&cpu_out, cpu_embedding_table, cpu_ids);
          LookupEmbedding<true>(&gpu_out, gpu_embedding_table, gpu_ids);
        }
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));

        {
          LookupEmbedding<false>(&cpu_out, cpu_embedding_table, cpu_ids);
          LookupEmbedding<false>(&gpu_out, gpu_embedding_table, gpu_ids);
        }
        REQUIRE(common::CheckResultOfCPUAndGPU<float>(cpu_out, gpu_out));
      }
}
#endif

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
