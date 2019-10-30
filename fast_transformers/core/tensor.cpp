#include "tensor.h"

namespace fast_transformers {
namespace core {
namespace details {
void DLManagedTensorDeletor(DLManagedTensor *self) {
  if (self == nullptr) {
    return;
  }
  if (self->dl_tensor.data != nullptr) {
    free(self->dl_tensor.data);
  }
  if (self->dl_tensor.shape != nullptr) {
    delete[] self->dl_tensor.shape;
  }
  delete self;
}
} // namespace details
} // namespace core
} // namespace fast_transformers
