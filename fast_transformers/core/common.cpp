#include "fast_transformers/core/common.h"

namespace fast_transformers {
namespace core {

bool is_same_device_ctx(DLContext t1, DLContext t2) {
  if (t1.device_id != t2.device_id || t1.device_type != t2.device_type) {
    return false;
  }
  return true;
}

bool is_same_shape(const Tensor &t1, const Tensor &t2) {
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

}  // namespace core
}  // namespace fast_transformers
