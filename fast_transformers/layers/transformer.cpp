#include "transformer.h"

namespace fast_transformers {
namespace kernels {
core::Tensor Transformer::operator()(const core::Tensor& c, const core::Tensor& attention_masks) const{
    //Tensor ->float*
    float* input_tensor_data = input_tensor.data();
    float* attr_mask_data = attention_masks.data();
    core::Tensor output_tensor;

    encoder_params_[0].from_tensor = input_tensor_data;
    encoder_params_[0].to_tensor   = input_tensor_data;
    for(int i = 0; i < num_layers_; ++i) {
        if(i > 0) {
            encoder_params_[i].from_tensor = bert_params_.layer_tensors[i-1];
            encoder_params_[i].to_tensor   = bert_params_.layer_tensors[i-1];
        }
        encoder_params_[i].transformer_out  = bert_params_.layer_tensors[i];
        encoder_params_[i].attr_mask = attr_mask_data;
        encoder_transformer_ops_[i]->initialize(encoder_params_[i]);
        encoder_transformer_ops_[i]->forward();
    }

    return core::Tensor(encoder_params_[num_layers_-1]);
}

} //kernels
} //fast_transformers