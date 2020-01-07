#define CATCH_CONFIG_MAIN
#include "prepare_bert_masks.h"

#include <chrono>

#include "catch2/catch.hpp"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/layers/kernels/test_helper.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {

#ifdef FT_WITH_CUDA
TEST_CASE("prepare_bert_masks CPU and GPU correctness") {
  std::vector<int64_t> batch_size_list{1, 20};
  std::vector<int64_t> seq_length_list{8, 16, 32, 48, 64, 128};
  for (auto batch_size : batch_size_list)
    for (auto seq_length : seq_length_list) {
      fast_transformers::core::Tensor gpu_inputs(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLGPU, 0));

      fast_transformers::core::Tensor cpu_inputs(
          fast_transformers::core::NewDLPackTensorT<float>(
              {batch_size, seq_length}, kDLCPU, 0));

      fast_transformers::core::Tensor* gpu_att_mask;
      fast_transformers::core::Tensor* cpu_att_mask;
      fast_transformers::core::Tensor* gpu_seq_type;
      fast_transformers::core::Tensor* cpu_seq_type;

      fast_transformers::core::Tensor* gpu_position_ids;
      fast_transformers::core::Tensor* cpu_position_ids;
      fast_transformers::core::Tensor* gpu_extended_attention_mask;
      fast_transformers::core::Tensor* cpu_extended_attention_mask;

      ::fast_transformers::test::FillDataForCPUGPUTensors<int64_t>(cpu_inputs,
                                                                   gpu_inputs);

      LOG_S(INFO) << "batch_size: " << batch_size
                  << " seq_length: " << seq_length;
      {
        PrepareBertMasks func;
        func(cpu_inputs, cpu_att_mask, cpu_seq_type, cpu_position_ids,
             cpu_extended_attention_mask);
        func(gpu_inputs, gpu_att_mask, gpu_seq_type, gpu_position_ids,
             gpu_extended_attention_mask);
      }
      REQUIRE(::fast_transformers::test::CompareCPUGPU<float>(*cpu_att_mask,
                                                              *gpu_att_mask));
      REQUIRE(::fast_transformers::test::CompareCPUGPU<float>(
          *cpu_extended_attention_mask, *gpu_extended_attention_mask));
      REQUIRE(::fast_transformers::test::CompareCPUGPU<int64_t>(*cpu_seq_type,
                                                                *gpu_seq_type));
      REQUIRE(::fast_transformers::test::CompareCPUGPU<int64_t>(
          *cpu_position_ids, *gpu_position_ids));
    }  // for
}
#endif

}  // namespace layers
}  // namespace fast_transformers
