#include "fast_transformers/core/device_context.h"

#include "absl/memory/memory.h"

namespace fast_transformers {
namespace core {

#ifdef WITH_CUDA

CUDADeviceContext::CUDADeviceContext(DLContext context) : context_(context) {
  cudaStreamCreate(&stream_);
  cublas_handle_.reset(new CublasHandleHolder(stream_));
}

void CUDADeviceContext::Wait() const {
  cudaError_t e_sync = cudaSuccess;
  e_sync = cudaStreamSynchronize(stream_);
  check_cuda_error(e_sync);
}

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

CUDADeviceContext::~CUDADeviceContext() {
  Wait();
  cublas_handle_.reset();
  check_cuda_error(cudaStreamDestroy(stream_));
}
#endif

template <typename DevCtx, DLDeviceType device_type>
inline void EmplaceDeviceContext(
    std::map<DLDeviceType, std::unique_ptr<DeviceContext>>*
        map_ptr,
        DLContext dl_ctx) {
  map_ptr->emplace(device_type, 
      absl::make_unique<DevCtx>(dl_ctx));
}

DeviceContextPool::DeviceContextPool(
    const std::vector<DLDeviceType>& device_types) {
  FT_ENFORCE_GT(device_types.size(), 0, "No device registed in DeviceContextPool");
  std::set<DLDeviceType> set;
  for (auto& t : device_types) {
    set.insert(t);
  }
  for (auto& t : set) {
    DLContext dl_ctx;
    if (t == kDLCPU) {
      dl_ctx.device_type = kDLCPU;
      EmplaceDeviceContext<CPUDeviceContext, kDLCPU>(&device_contexts_, dl_ctx);
    } else if (t == kDLGPU) {
#ifdef WITH_CUDA
      dl_ctx.device_type = kDLGPU;
      dl_ctx.device_id = 0;
      EmplaceDeviceContext<CUDADeviceContext, kDLGPU>(&device_contexts_, dl_ctx);
#else
#endif
    }
  }
}

DeviceContext* DeviceContextPool::Get(const DLDeviceType device_type) {
  auto it = device_contexts_.find(device_type);
  if (it == device_contexts_.end()) {
    FT_THROW(
        "device_type %d is not supported", device_type);
  }
  return it->second.get();
}

}  // namespace core
}  // namespace fast_transformers
