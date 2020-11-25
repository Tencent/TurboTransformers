

#include "common.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
namespace common {

bool is_same_device_ctx(DLContext t1, DLContext t2) {
  return t1.device_id == t2.device_id && t1.device_type == t2.device_type;
}

bool is_same_shape(const core::Tensor& t1, const core::Tensor& t2) {
  if (t1.n_dim() != t2.n_dim()) {
    return false;
  }
  for (size_t i = 0; i < t1.n_dim(); ++i) {
    if (t1.shape(i) != t2.shape(i)) {
      return false;
    }
  }
  return true;
}

template <typename T>
void Sequence(T* data, int64_t size, DLDeviceType device) {
  if (device == kDLCPU) {
    std::iota(data, data + size, static_cast<T>(0));
  } else if (device == kDLGPU) {
#ifdef TT_WITH_CUDA
    turbo_transformers::layers::kernels::GPUSequence(data, size);
#else
    TT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}
template void Sequence(float* data, int64_t size, DLDeviceType device);
template void Sequence(int64_t* data, int64_t size, DLDeviceType device);

template <typename T>
void Fill(T* data, int64_t size, T val, DLDeviceType device) {
  if (device == kDLCPU) {
    std::fill(data, data + size, val);
  } else if (device == kDLGPU) {
#ifdef TT_WITH_CUDA
    layers::kernels::GPUFill(data, size, val);
#else
    TT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}

template void Fill<float>(float* data, int64_t size, float val,
                          DLDeviceType device);
template void Fill<int64_t>(int64_t* data, int64_t size, int64_t val,
                            DLDeviceType device);

// TODO(jiaruifang): this function should better pass a function in.
// how can we pass a lambda function as __device__ to cuda?
void Transform(int64_t* src_data, float* dst_data, int64_t size,
               DLDeviceType device) {
  if (device == kDLCPU) {
    std::transform(src_data, src_data + size, dst_data,
                   [](int64_t v) { return -10000.0f * (1 - v); });
  } else if (device == kDLGPU) {
#ifdef TT_WITH_CUDA
    layers::kernels::GPUTransform(src_data, dst_data, size);
#else
    TT_THROW("code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}

}  // namespace common
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
