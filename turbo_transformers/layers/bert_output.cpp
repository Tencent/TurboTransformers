

#include "turbo_transformers/layers/bert_output.h"

#include <loguru.hpp>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {

void BertOutput::operator()(const core::Tensor &hidden_states,
                            const core::Tensor &input_tensor,
                            core::Tensor *output_tensor) const {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile("BertOutput", input_tensor.device_type());
#endif
  TT_ENFORCE_EQ(kernels::common::is_same_device_ctx(input_tensor.device_ctx(),
                                                    hidden_states.device_ctx()),
                true,
                "BertOutput: The input_tensor and hidden_states should have "
                "the same device type and device id.");
  output_tensor->Reshape<float>(
      {hidden_states.shape(0), hidden_states.shape(1), dense_weight_.shape(1)},
      hidden_states.device_type(), hidden_states.device_id(),
      "BERTEmbedding/Reshape");
  // NOTE the out of this bert layer should be the input of the next layer
  //      "BertOutput/Reshape");
  kernels::MatMul(hidden_states, false, dense_weight_, false, 1.0,
                  output_tensor, 0.0, "BertOutput/MatMul");
  kernels::AddBiasLayerNorm<float>(
      input_tensor, dense_bias_, layer_norm_weight_, layer_norm_bias_,
      output_tensor, 1e-12, "BertOutput/AddBiasLayerNorm");
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile("BertOutput", input_tensor.device_type());
#endif
}

void BertOutput::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::stringstream ss;
    ss << "<<<<<<<< dense_weight_ <<<<<<<<<<";
    dense_weight_.Print<float>(ss);
    ss << "<<<<<<<< dense_bias <<<<<<<<<<";
    dense_bias_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_weight <<<<<<<<<<";
    layer_norm_weight_.Print<float>(ss);
    ss << "<<<<<<<< layer_norm_bias <<<<<<<<<<";
    layer_norm_bias_.Print<float>(ss);
    LOG_S(3) << ss.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
