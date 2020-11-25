

#pragma once

#include <vector>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/core/tensor.h"

#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace core {
template <typename T>
static inline void Copy(const T *data, size_t size, DLDeviceType srcDevice,
                        DLDeviceType dstDevice, T *dst) {
  auto flag = core::ToMemcpyFlag(dstDevice, srcDevice);
  core::Memcpy(dst, data, sizeof(T) * size, flag);
}
template <typename T>
static inline void Copy(const T *data, size_t size, DLDeviceType srcDevice,
                        core::Tensor &dst) {
  DLDeviceType dstDevice = dst.device_type();
  auto flag = core::ToMemcpyFlag(dstDevice, srcDevice);
  core::Memcpy(dst.mutableData<T>(), data, sizeof(T) * size, flag);
}
template <typename T>
static inline void Copy(const core::Tensor &src, std::vector<T> &dst) {
  auto flag = core::ToMemcpyFlag(DLDeviceType::kDLCPU, src.device_type());
  core::Memcpy(dst.data(), src.data<T>(), sizeof(T) * src.numel(), flag);
}
template <typename T>
static inline void Copy(const core::Tensor &src, core::Tensor &dst,
                        const std::string name = "Copy") {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, src.device_type());
#endif
  TT_ENFORCE_EQ(dst.numel(), src.numel(),
                "Copy two tensors should have the same size");
  auto flag = core::ToMemcpyFlag(dst.device_type(), src.device_type());
  core::Memcpy(dst.mutableData<T>(), src.data<T>(), sizeof(T) * src.numel(),
               flag);
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, src.device_type());
#endif
}

}  // namespace core
}  // namespace turbo_transformers
