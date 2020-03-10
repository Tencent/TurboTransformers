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

#include "fast_transformers/core/memory.h"

#include <cstring>
#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_error.h"
#endif

namespace fast_transformers {
namespace core {
void* align_alloc(size_t sz, size_t align) {
  void* aligned_mem;
  FT_ENFORCE_EQ(posix_memalign(&aligned_mem, align, sz), 0,
                "Cannot allocate align memory with %d bytes, "
                "align %d",
                sz, align);
  return aligned_mem;
}

void FT_Memcpy(void* dst_data, const void* src_data, size_t data_size,
               MemcpyFlag flag) {
  if (data_size <= 0) return;
  if (flag == MemcpyFlag::kCPU2CPU) {
    std::memcpy(dst_data, src_data, data_size);
  }
#ifdef FT_WITH_CUDA
  else if (flag == MemcpyFlag::kGPU2CPU) {
    FT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(((void*)dst_data), ((void*)src_data),
                                       data_size, cudaMemcpyDeviceToHost));
  } else if (flag == MemcpyFlag::kCPU2GPU) {
    FT_ENFORCE_CUDA_SUCCESS(cudaMemcpy(((void*)dst_data), ((void*)src_data),
                                       data_size, cudaMemcpyHostToDevice));
  }
#endif
  else {
    FT_THROW("The current MemcpyFlag is not support now.");
  }
}

}  // namespace core
}  // namespace fast_transformers
