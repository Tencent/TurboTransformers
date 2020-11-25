

#include "turbo_transformers/layers/bert_attention.h"

#include <unordered_map>

#include "loguru.hpp"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"

namespace turbo_transformers {
namespace layers {

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output, core::Tensor* attn,
                               bool is_trans_weight) const {
  std::unordered_map<std::string, core::Tensor*> dummy{};
  core::Tensor* attn_ptr;
  if (attn == nullptr) {
    attn_ptr = new core::Tensor(nullptr);
  } else {
    attn_ptr = attn;
  }
  MultiHeadedAttention::operator()(
      input_tensor, input_tensor, input_tensor, attention_mask, "self", output,
      attn_ptr, dummy, false /* pre_layernorm */, true /* post_layernorm */,
      false /* post_add_input */, is_trans_weight /* is_trans_weight */);
  if (attn == nullptr) {
    delete attn_ptr;
  }
}

void BertAttention::EnforceShapeAndType() const {
  MultiHeadedAttention::EnforceShapeAndType();
}

}  // namespace layers
}  // namespace turbo_transformers
