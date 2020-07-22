// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.
#include "turbo_transformers/layers/kernels/embedding.h"

#include "common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_embedding_kernel.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool Add>
void LookupEmbedding(core::Tensor *out_tensor,
                     const core::Tensor &embedding_table,
                     const core::Tensor &ids_tensor,
                     const std::string name) {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, ids_tensor.device_type());
#endif
  TT_ENFORCE_EQ(common::is_same_device_ctx(
                    out_tensor->device_ctx(), embedding_table.device_ctx()),
                true,
                "The out_tensor and embedding_table should have the same "
                "device type and device id.");

  TT_ENFORCE_EQ(common::is_same_device_ctx(out_tensor->device_ctx(),
                                           ids_tensor.device_ctx()),
                true,
                "The out_tensor and ids_tensor should have the same device "
                "type and device id.");

  const float *embedding = embedding_table.data<float>();
  const int64_t *ids = ids_tensor.data<int64_t>();
  auto *out = out_tensor->mutableData<float>();
  auto num_ids = ids_tensor.numel();
  auto hidden_size = embedding_table.shape(1);
  auto vocab_size = embedding_table.shape(0);
  if (out_tensor->device_type() == kDLCPU) {
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
  } else if (out_tensor->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto &cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPULookupKernel<Add>(out, embedding, ids, vocab_size, hidden_size,
                         num_ids, cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, ids_tensor.device_type());
#endif
}

template void LookupEmbedding<true>(core::Tensor *out_tensor,
                                    const core::Tensor &embedding_table,
                                    const core::Tensor &ids_tensor,
                                    const std::string name);

template void LookupEmbedding<false>(core::Tensor *out_tensor,
                                     const core::Tensor &embedding_table,
                                     const core::Tensor &ids_tensor,
                                     const std::string name);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
