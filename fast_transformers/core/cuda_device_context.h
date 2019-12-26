#pragma once
#include <memory.h>

#include "fast_transformers/core/cuda_error.h"
#include "macros.h"

namespace fast_transformers {
namespace core {

class CublasHandleHolder {
 public:
  CublasHandleHolder(cudaStream_t stream) {
    cublasCreate(&handle_);
    cublasSetStream(handle_, stream);
  }

  ~CublasHandleHolder() noexcept(false) { cublasDestroy(handle_); }

  template <typename Callback>
  inline void Call(Callback&& callback) const {
    callback(handle_);
  }

 private:
  DISABLE_COPY_AND_ASSIGN(CublasHandleHolder);

  cublasHandle_t handle_;
};

/*
Initialize DeviceContext with DLContext, although currently DLContext is not
defined.
 */
class CUDADeviceContext {
 public:
  explicit CUDADeviceContext();
  virtual ~CUDADeviceContext();

  static CUDADeviceContext& GetInstance() {
    static CUDADeviceContext instance;
    return instance;
  }

  void Wait() const;

  template <typename Callback>
  inline void CublasCall(Callback&& callback) const {
    cublas_handle_->Call(std::forward<Callback>(callback));
  }

  cudaStream_t stream() const;
  int device_count() const;

 private:
  cudaStream_t stream_;
  std::unique_ptr<CublasHandleHolder> cublas_handle_;

  DISABLE_COPY_AND_ASSIGN(CUDADeviceContext);
};

}  // namespace core
}  // namespace fast_transformers
