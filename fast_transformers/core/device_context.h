#pragma once

#include <vector>
#include <map>
#include <set>
#include "macros.h"
#include "fast_transformers/core/enforce.h"
#include <dlpack/dlpack.h>

#ifdef WITH_CUDA
#include "fast_transformers/core/nvcommon.h"
#endif

namespace fast_transformers {
namespace core {

class DeviceContext { 
 public:
  virtual ~DeviceContext() noexcept(false) {}

  virtual void Wait() const {}
};

template <DLDeviceType Device>
struct DefaultDeviceContextType;

class CPUDeviceContext : public DeviceContext {
 public:
  explicit CPUDeviceContext(DLContext context) : context_(context) {}

 private:
  DLContext context_;
};

template <>
struct DefaultDeviceContextType<kDLCPU> {
  using TYPE = CPUDeviceContext;
};

#ifdef WITH_CUDA
class CublasHandleHolder {
 public:
  CublasHandleHolder(cudaStream_t stream) {
    cublasCreate(&handle_);
    cublasSetStream(handle_, stream);
  }

  ~CublasHandleHolder() noexcept(false) {
    cublasDestroy(handle_);
  }

  template <typename Callback>
  inline void Call(Callback &&callback) const {
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  cublasHandle_t handle_;
};

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(DLContext context);
  virtual ~CUDADeviceContext();

  void Wait() const override;

  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    cublas_handle_->Call(std::forward<Callback>(callback));
  }

  cudaStream_t stream() const;

 private:
  DLContext context_;

  cudaStream_t stream_;
  std::unique_ptr<CublasHandleHolder> cublas_handle_;

  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};


template <>
struct DefaultDeviceContextType<kDLGPU> {
  using TYPE = CUDADeviceContext;
};

#endif

/*
 * Now we can only have one CPU and One GPU
 * a.k.a each device type has only one device
 * DeviceContextPool pool = DeviceContextPool::Instance();
 * gpu_cxt = pool.Get(kDLGPU);
 */
class DeviceContextPool {
 public:
  explicit DeviceContextPool(const std::vector<DLDeviceType>& devices);

  // init singleton in hungrey pattern
  static DeviceContextPool& Instance() {
#ifdef WITH_CUDA
  	static DeviceContextPool pool({kDLCPU, kDLGPU});
#else
	static DeviceContextPool pool({kDLCPU});
#endif
    return pool;
  }

  DeviceContext* Get(const DLDeviceType device_type);

  template <DLDeviceType device_type>
  const typename DefaultDeviceContextType<device_type>::TYPE* GetByDLDeviceType(
      const DLDeviceType& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<device_type>::TYPE*>(Get(place));
  }

  size_t size() const { return device_contexts_.size(); }

 private:
  std::map<DLDeviceType, std::unique_ptr<DeviceContext>>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace core
}  // namespace fast_transformers
