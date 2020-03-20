#pragma once

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace core {
template <typename T>
static inline void Copy(core::Tensor &dst, const T *data, size_t size,
                        DLDeviceType srcDevice) {
  DLDeviceType dstDevice = dst.device_type();
  auto flag = core::ToMemcpyFlag(dstDevice, srcDevice);
  core::FT_Memcpy(dst.mutableData<T>(), data, sizeof(T) * size, flag);
}
template <typename T>
static inline void Copy(std::vector<T> &dst, core::Tensor &src) {
  auto flag = core::ToMemcpyFlag(DLDeviceType::kDLCPU, src.device_type());
  core::FT_Memcpy(dst.data(), src.data<T>(), sizeof(T) * src.numel(), flag);
}

}  // namespace core
}  // namespace turbo_transformers
