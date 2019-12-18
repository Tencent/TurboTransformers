#include "tensor.h"

namespace fast_transformers {
namespace core {
static void DLManagedTensorDeletor(DLManagedTensor *self) {
  if (self == nullptr) {
    return;
  }
  if (self->dl_tensor.data != nullptr) {
    if (self->dl_tensor.ctx.device_type == kDLCPU) {
      free(self->dl_tensor.data);
    } else if (self->dl_tensor.ctx.device_type == kDLGPU) {
#ifdef FT_WITH_CUDA
      cuda_free(self->dl_tensor.data);
#endif
    }
  }

  delete[] self->dl_tensor.shape;
  delete self;
}

DLManagedTensor *NewDLPackTensor(std::initializer_list<int64_t> shape_list,
                                 DLDeviceType device, int device_id,
                                 uint8_t data_type_code, size_t bits,
                                 size_t lanes) {
  FT_ENFORCE_NE(shape_list.size(), 0, "Shape list should not be empty");
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
  if (device == kDLCPU) {
    newTensor->dl_tensor.data = align_alloc(numel * (bits / 8));
  } else if (device == kDLGPU) {
#ifdef FT_WITH_CUDA
    newTensor->dl_tensor.data = cuda_alloc(numel * (bits / 8));
#endif
  } else {
    FT_THROW("only cpu and gpu are supported!");
  }

  newTensor->deleter = DLManagedTensorDeletor;
  return newTensor;
}

}  // namespace core
}  // namespace fast_transformers
