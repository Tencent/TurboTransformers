#include "prepare_bert_masks.h"
#include <stdint.h>
#include "fast_transformers/core/eigen-tensor.h"
namespace fast_transformers {
namespace layers {
void PrepareBertMasks::operator()(const core::Tensor& inputs,
                                  core::Tensor* att_mask,
                                  core::Tensor* seq_type,
                                  core::Tensor* position_ids,
                                  core::Tensor* extended_attention_mask) const {
  if (position_ids->is_null()) {
    auto pos_ids_ptr =
        position_ids->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)});

    // fill range
    for (int64_t row_id = 0; row_id < inputs.shape(0); ++row_id) {
      std::iota(pos_ids_ptr, pos_ids_ptr + inputs.shape(1), 0);
      pos_ids_ptr += inputs.shape(1);
    }
  }

  if (seq_type->is_null()) {
    // seq_type.zeros_like(inputs)
    seq_type->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)});
    core::to_tensor<2, int64_t>(seq_type).setZero();
  }

  if (att_mask->is_null()) {
    att_mask->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)});
    core::to_tensor<2, int64_t>(att_mask).setConstant(1);
  }

  // cast att_mask to float
  extended_attention_mask->Reshape<float>(
      {att_mask->shape(0), 1, 1, att_mask->shape(1)});

  auto att_mask_tensor = core::to_tensor<2, int64_t>(att_mask);
  auto extended_attention_mask_tensor =
      core::to_tensor<4, float>(extended_attention_mask);

  extended_attention_mask_tensor.device(core::CPUDevice()) =
      (1 - att_mask_tensor.reshape(Eigen::array<int, 4>{
               (int)att_mask->shape(0), 1, 1, (int)att_mask->shape(1)}))
          .cast<float>() *
      (-10000.0f);
}
}  // namespace layers
}  // namespace fast_transformers
