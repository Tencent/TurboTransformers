// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo_transformers/layers/bert_embedding.h"

#include "loguru.hpp"
#include "turbo_transformers/core/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_embedding_kernel.h"
#endif

namespace turbo_transformers {
namespace layers {

template <bool Add>
static void LookupEmbedding(core::Tensor &out_tensor,
                            const core::Tensor &embedding_table,
                            const core::Tensor &ids_tensor) {
  TT_ENFORCE_EQ(core::common::is_same_device_ctx(out_tensor.device_ctx(),
                                                 embedding_table.device_ctx()),
                true,
                "The out_tensor and embedding_table should have the same "
                "device type and device id.");

  TT_ENFORCE_EQ(core::common::is_same_device_ctx(out_tensor.device_ctx(),
                                                 ids_tensor.device_ctx()),
                true,
                "The out_tensor and ids_tensor should have the same device "
                "type and device id.");

  const float *embedding = embedding_table.data<float>();
  const int64_t *ids = ids_tensor.data<int64_t>();
  auto *out = out_tensor.mutableData<float>();
  auto num_ids = ids_tensor.numel();
  auto hidden_size = embedding_table.shape(1);
  auto vocab_size = embedding_table.shape(0);
  if (out_tensor.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < num_ids; ++i) {
      int64_t id = ids[i];
      TT_ENFORCE_LT(id, vocab_size, "embedding id out of index");
      auto dst = out + i * hidden_size;
      auto src = embedding + id * hidden_size;
      if (Add) {
#pragma omp simd
        for (int64_t j = 0; j < hidden_size; ++j) {
          dst[j] += src[j];
        }
      } else {
        std::copy(src, src + hidden_size, dst);
      }
    }
  } else if (out_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto &cuda_ctx = core::CUDADeviceContext::GetInstance();
    kernels::GPULookupKernel<Add>(out, embedding, ids, vocab_size, hidden_size,
                                  num_ids, cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}

void BERTEmbedding::operator()(const core::Tensor &input_ids,
                               const core::Tensor &position_ids,
                               const core::Tensor &token_type_ids,
                               core::Tensor *output_tensor) const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> input_ids <<<<<<<<<<<<" << std::endl;
    input_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> position_ids <<<<<<<<<<<<" << std::endl;
    position_ids.Print<int64_t>(os);
    os << ">>>>>>>>>>>> token_type_ids <<<<<<<<<<<<" << std::endl;
    token_type_ids.Print<int64_t>(os);
    LOG_S(3) << os.str();
  }

  TT_ENFORCE_EQ(
      input_ids.n_dim(), 2,
      "The input ids should be a matrix with shape [BatchSize, SeqLen].");
  auto batch_size = input_ids.shape(0);
  auto seq_length = input_ids.shape(1);
  // TODO 1. switch DeviceType::CPU 2. how should I set stride?
  auto hidden_size = word_embedings_.shape(1);

  TT_ENFORCE(output_tensor, "The output tensor should not be nullptr.");
  output_tensor->Reshape<float>({batch_size, seq_length, hidden_size},
                                input_ids.device_type(), input_ids.device_id());
  LOG_S(3) << "Look up word embedding";
  LookupEmbedding</*Add=*/false>(*output_tensor, word_embedings_, input_ids);
  LOG_S(3) << "Look up token type embedding";
  LookupEmbedding</*Add=*/true>(*output_tensor, token_type_embeddings_,
                                token_type_ids);
  LOG_S(3) << "Look up token position embedding";
  LookupEmbedding</*Add=*/true>(*output_tensor, position_embeddings_,
                                position_ids);

  kernels::LayerNorm<float>(layer_norm_weights_, layer_norm_bias_,
                            output_tensor);
}
void BERTEmbedding::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>> word_embedings_ <<<<<<<<" << std::endl;
    word_embedings_.Print<float>(os);
    os << ">>>>> position_embeddings_ <<<<<<<<" << std::endl;
    position_embeddings_.Print<float>(os);
    os << ">>>>> token_type_embeddings_ <<<<<<<<" << std::endl;
    token_type_embeddings_.Print<float>(os);
    os << ">>>>> layer_norm_weights_ <<<<<<<<" << std::endl;
    layer_norm_weights_.Print<float>(os);
    os << ">>>>> layer_norm_bias_ <<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}
}  // namespace layers
}  // namespace turbo_transformers
