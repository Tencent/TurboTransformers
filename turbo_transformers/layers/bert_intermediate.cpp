

#include "turbo_transformers/layers/bert_intermediate.h"

#include <loguru.hpp>

#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {

void BertIntermediate::operator()(const core::Tensor& input_tensor,
                                  core::Tensor* output_tensor) const {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("BertIntermediate", input_tensor.device_type());
#endif
  output_tensor->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id(),
      "BertIntermediate/Reshape");

  kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, output_tensor,
                  0.0, "BertIntermediate/MatMul");
  kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(
      dense_bias_, output_tensor, "BertIntermediate/AddBiasAct");
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("BertIntermediate", input_tensor.device_type());
#endif
}

void BertIntermediate::EnforceShapeAndType() const {
  TT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  TT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  TT_ENFORCE_EQ(dense_weight_.shape(1), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(1),
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
}  // namespace turbo_transformers
