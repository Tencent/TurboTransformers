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

#include "tensor.h"

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#endif
#include "turbo_transformers/core/allocator.h"

namespace turbo_transformers {
namespace core {
static void DLManagedTensorDeletor(DLManagedTensor *self) {
  if (self == nullptr) {
    return;
  }
  if (self->dl_tensor.data != nullptr) {
    if (self->dl_tensor.ctx.device_type == kDLCPU ||
        self->dl_tensor.ctx.device_type == kDLGPU) {
      // free(self->dl_tensor.data);
      Allocator &allocator = Allocator::GetInstance();
      allocator.free(self->dl_tensor.data, "cub",
                     self->dl_tensor.ctx.device_type);
    }
  }

  delete[] self->dl_tensor.shape;
  delete self;
}

DLManagedTensor *NewDLPackTensor(const std::vector<int64_t> &shape_list,
                                 DLDeviceType device, int device_id,
                                 uint8_t data_type_code, size_t bits,
                                 size_t lanes) {
  TT_ENFORCE_NE(shape_list.size(), 0, "Shape list should not be empty");
  auto *newTensor = new DLManagedTensor();

  newTensor->dl_tensor.shape = new int64_t[shape_list.size()];
  std::copy(shape_list.begin(), shape_list.end(), newTensor->dl_tensor.shape);

  newTensor->dl_tensor.ctx = {device, device_id};  // device_type, device_id
  newTensor->dl_tensor.ndim = shape_list.size();

  newTensor->dl_tensor.dtype = {
      static_cast<uint8_t>(data_type_code), static_cast<uint8_t>(bits),
      static_cast<uint16_t>(lanes)};  // code, bits, lanes

  newTensor->dl_tensor.strides = nullptr;  // TODO
  newTensor->dl_tensor.byte_offset = 0;

  size_t numel = std::accumulate(shape_list.begin(), shape_list.end(), 1,
                                 std::multiplies<int64_t>());
  if (device == kDLCPU || device == kDLGPU) {
    size_t size = numel * (bits / 8);
    Allocator &allocator = Allocator::GetInstance();
    newTensor->dl_tensor.data = allocator.allocate(size, "cub", device);
  } else {
    TT_THROW("only cpu and gpu are supported!");
  }

  newTensor->deleter = DLManagedTensorDeletor;
  return newTensor;
}

}  // namespace core
}  // namespace turbo_transformers
