#include "fast_transformers/layers/bert_intermediate.h"

#include <loguru.hpp>

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/activation.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/mat_mul.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"

namespace fast_transformers {
namespace layers {

void BertIntermediate::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor) const {
  output_tensor->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(0)},
      input_tensor.device_type());

  core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
  kernels::MatMul(input_tensor, false, dense_weight_, true, 1.0, output_tensor,
                  0.0);
  kernels::AddBiasGeLUAct<float>(dense_bias_, output_tensor);
}

void BertIntermediate::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  FT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  FT_ENFORCE_EQ(dense_weight_.shape(0), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(0),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
