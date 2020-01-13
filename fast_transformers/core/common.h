#pragma once
#include <dlpack/dlpack.h>
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace core {

extern bool is_same_device_ctx(DLContext t1, DLContext t2);

extern bool is_same_shape(const Tensor &t1, const Tensor &t2);

}  // namespace core
}  // namespace fast_transformers
